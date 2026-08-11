#ifndef DFKV_MEMBERSHIP_H_
#define DFKV_MEMBERSHIP_H_

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "utils/net_util.h"  // net::PutU32/PutU64/GetU32/GetU64 (host-endian codec)

namespace dfkv {

// Dynamic runtime stats a node reports on register/heartbeat (STA1 trailing
// extension, ~10s freshness). Fixed field ORDER is the wire contract -- extend
// by APPENDING (the on-wire field count lets old decoders skip unknown tails).
// All counters are per-node monotonic since process start; uptime_seconds is
// the reset anchor (a restart zeroes them). Excluded from operator== and
// MembersEpoch like every other extension: stats churn every heartbeat and
// must never rebuild client rings.
struct MemberStats {
  uint64_t capacity_bytes = 0;        // configured cap (static; binary twin of info cap=)
  uint64_t used_bytes = 0;            // engine-reported (slab: slot bytes; file: payload bytes)
  uint64_t objects = 0;
  uint64_t hits_total = 0;
  uint64_t misses_total = 0;
  uint64_t evictions_total = 0;
  uint64_t puts_total = 0;
  uint64_t uptime_seconds = 0;
  uint64_t put_busy_total = 0;        // admission-gate rejections
  uint64_t dio_write_fallbacks = 0;   // direct-mode buffered fallbacks ("page cache crept back")
  uint64_t ram_used_bytes = 0;        // RAM tier resident slot bytes (0 = tier off)
  uint64_t ram_hits_total = 0;
};
constexpr uint32_t kMemberStatsFields = 12;  // == u64 count above, wire order

struct IbDeviceHealth {
  std::string name;
  uint8_t port_state = 0;
  uint8_t phys_state = 0;
  bool query_ok = false;

  bool healthy() const {
    return query_ok && port_state == 4 && phys_state == 5;
  }
};

struct MemberHealth {
  bool ring_eligible = true;
  std::vector<IbDeviceHealth> ib_devices;
};

inline const char* IbPortStateName(uint8_t state) {
  switch (state) {
    case 1: return "DOWN";
    case 2: return "INIT";
    case 3: return "ARMED";
    case 4: return "ACTIVE";
    case 5: return "ACTIVE_DEFER";
    default: return "UNKNOWN";
  }
}

inline const char* IbPhysStateName(uint8_t state) {
  switch (state) {
    case 1: return "SLEEP";
    case 2: return "POLLING";
    case 3: return "DISABLED";
    case 4: return "TRAINING";
    case 5: return "LINK_UP";
    case 6: return "ERROR_RECOVERY";
    case 7: return "PHY_TEST";
    default: return "UNKNOWN";
  }
}

struct MemberInfo {
  std::string id;
  std::string ip;
  uint32_t port = 0;       // data-path port clients dial (the rdma-port in an RDMA deploy)
  uint32_t weight = 1;
  uint32_t tcp_port = 0;   // TCP wire/stat port that serves kStats; 0 = unknown / older peer.
  // Node self-description reported on register/heartbeat: "k=v,k=v,..." (version,
  // storage engine, capacity, RAM tier, RDMA dev ...). Empty = unknown / older peer.
  // Purely informational: surfaced by `dfkvctl ring` for fleet audit (version skew,
  // silently-skipped upgrades, engine mismatch) without per-node ssh.
  std::string info{};
  // Dynamic stats (STA1); has_stats=false = legacy peer / no report yet.
  // LAST members on purpose: MemberInfo is aggregate-initialized positionally
  // in tests/tools, and appending keeps those initializers valid. The explicit
  // {} initializers keep those shorter positional inits warning-free
  // (-Wmissing-field-initializers skips NSDMI'd members).
  MemberStats stats{};
  bool has_stats = false;
  MemberHealth health{};
  bool has_health = false;
  // tcp_port and info ride OPTIONAL trailing extensions in Encode/DecodeMembers, so peers
  // that don't send them still interoperate. Both are intentionally EXCLUDED from
  // operator== and MembersEpoch: they are orthogonal metadata (not part of ring
  // placement), and their consumers (dfkvctl) re-fetch every run, so a change need not
  // trigger a client ring rebuild.
  bool RingEligible() const {
    return !has_health || health.ring_eligible;
  }
  bool operator==(const MemberInfo& o) const {
    return id == o.id && ip == o.ip && port == o.port && weight == o.weight &&
           RingEligible() == o.RingEligible();
  }
};

// A group or member id is embedded verbatim into the etcd key
//   /dfkv/v1/groups/<group>/members/<id>
// so a '/' (or an empty/overlong token) lets a registration escape its own
// key subtree -- e.g. group "a/members/ghost" injects a phantom member into
// group "a"'s RangePrefix, and any reachable client can point a real node id
// at an attacker IP. Restrict both to a conservative, path-safe alphabet.
inline bool IsValidGroupOrId(const std::string& s) {
  if (s.empty() || s.size() > 128) return false;
  for (unsigned char c : s) {
    const bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                    (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-';
    if (!ok) return false;
  }
  return true;
}

// Magic tags marking OPTIONAL extensions appended after the member list, in tag
// order (TCP1 then NFO1). Old encodings carry no trailing bytes and old decoders
// stop after the member list, so the tags are unambiguous when present and
// invisible to peers that predate them.
constexpr uint32_t kMemberExtTcpPort = 0x54435031u;  // "TCP1": one tcp_port u32 per member
constexpr uint32_t kMemberExtInfo = 0x4E464F31u;     // "NFO1": one length-prefixed info string per member
constexpr uint32_t kMemberExtStats = 0x31415453u;    // "STA1": nfields u8 then per member { has u8 [nfields x u64] }
constexpr uint32_t kMemberExtHealth = 0x31544C48u;   // "HLT1": eligibility + per-device IB health

// Wire format for a membership view. Register payload = 1 member; ListMembers
// response = N members + the epoch (etcd revision) clients compare to skip
// rebuilding an unchanged ring. Layout (host-endian via net::):
//   epoch u64 | count u32 | repeat{ idlen u32, id, iplen u32, ip, port u32, weight u32 }
//   [ optional: kMemberExtTcpPort u32 | repeat count { tcp_port u32 } ]        <- old decoders ignore
//   [ optional: kMemberExtInfo u32 | repeat count { infolen u32, info bytes } ] <- old decoders ignore
// (host-endian via net::; correct between same-endianness peers — see net_util.h)
inline std::string EncodeMembers(const std::vector<MemberInfo>& ms,
                                 uint64_t epoch) {
  std::string out;
  char hdr[12];
  net::PutU64(hdr, epoch);
  net::PutU32(hdr + 8, static_cast<uint32_t>(ms.size()));
  out.append(hdr, 12);
  char num[4];
  for (const auto& m : ms) {
    net::PutU32(num, static_cast<uint32_t>(m.id.size())); out.append(num, 4);
    out += m.id;
    net::PutU32(num, static_cast<uint32_t>(m.ip.size())); out.append(num, 4);
    out += m.ip;
    net::PutU32(num, m.port);   out.append(num, 4);
    net::PutU32(num, m.weight); out.append(num, 4);
  }
  // Optional trailing extensions: a magic tag + per-member payload, same order.
  // Older decoders stop after the N members above and never read these bytes.
  net::PutU32(num, kMemberExtTcpPort); out.append(num, 4);
  for (const auto& m : ms) { net::PutU32(num, m.tcp_port); out.append(num, 4); }
  net::PutU32(num, kMemberExtInfo); out.append(num, 4);
  for (const auto& m : ms) {
    net::PutU32(num, static_cast<uint32_t>(m.info.size())); out.append(num, 4);
    out += m.info;
  }
  // STA1: dynamic stats. nfields on the wire (not a version) so appending
  // fields later needs no new tag: an older decoder reads the fields it knows
  // and skips the tail. Per-member `has` flag: a kList response mixes members
  // from different node versions, so presence is per member, not per message.
  net::PutU32(num, kMemberExtStats); out.append(num, 4);
  out.push_back(static_cast<char>(kMemberStatsFields));
  char u64buf[8];
  for (const auto& m : ms) {
    out.push_back(m.has_stats ? 1 : 0);
    if (!m.has_stats) continue;
    const uint64_t f[kMemberStatsFields] = {
        m.stats.capacity_bytes, m.stats.used_bytes, m.stats.objects,
        m.stats.hits_total, m.stats.misses_total, m.stats.evictions_total,
        m.stats.puts_total, m.stats.uptime_seconds, m.stats.put_busy_total,
        m.stats.dio_write_fallbacks, m.stats.ram_used_bytes, m.stats.ram_hits_total};
    for (uint32_t k = 0; k < kMemberStatsFields; ++k) {
      net::PutU64(u64buf, f[k]);
      out.append(u64buf, 8);
    }
  }
  if (std::any_of(ms.begin(), ms.end(),
                  [](const MemberInfo& m) { return m.has_health; })) {
    net::PutU32(num, kMemberExtHealth); out.append(num, 4);
    for (const auto& m : ms) {
      out.push_back(m.has_health ? 1 : 0);
      if (!m.has_health) continue;
      out.push_back(m.health.ring_eligible ? 1 : 0);
      net::PutU32(num, static_cast<uint32_t>(m.health.ib_devices.size()));
      out.append(num, 4);
      for (const auto& d : m.health.ib_devices) {
        net::PutU32(num, static_cast<uint32_t>(d.name.size()));
        out.append(num, 4);
        out += d.name;
        out.push_back(static_cast<char>(d.port_state));
        out.push_back(static_cast<char>(d.phys_state));
        out.push_back(d.query_ok ? 1 : 0);
      }
    }
  }
  return out;
}

// Returns false on truncated / malformed input (caller treats as a failed RPC).
inline bool DecodeMembers(const char* p, size_t n,
                          std::vector<MemberInfo>* out, uint64_t* epoch) {
  if (n < 12) return false;
  *epoch = net::GetU64(p);
  uint32_t count = net::GetU32(p + 8);
  size_t off = 12;
  out->clear();
  for (uint32_t i = 0; i < count; ++i) {
    MemberInfo m;
    if (off + 4 > n) return false;
    uint32_t idlen = net::GetU32(p + off); off += 4;
    if (off + idlen > n) return false;
    m.id.assign(p + off, idlen); off += idlen;
    if (off + 4 > n) return false;
    uint32_t iplen = net::GetU32(p + off); off += 4;
    if (off + iplen > n) return false;
    m.ip.assign(p + off, iplen); off += iplen;
    if (off + 8 > n) return false;
    m.port = net::GetU32(p + off);   off += 4;
    m.weight = net::GetU32(p + off);  off += 4;
    out->push_back(std::move(m));
  }
  // Optional trailing extensions (see EncodeMembers): read tagged blocks in
  // order; an unknown tag or a truncated block ends the (best-effort) ext scan
  // without failing the decode -- exts absent from older peers' encodings just
  // leave tcp_port==0 / info=="".
  while (off + 4 <= n) {
    const uint32_t tag = net::GetU32(p + off);
    if (tag == kMemberExtTcpPort) {
      off += 4;
      if (off + static_cast<size_t>(count) * 4 > n) break;
      for (uint32_t i = 0; i < count; ++i) { (*out)[i].tcp_port = net::GetU32(p + off); off += 4; }
    } else if (tag == kMemberExtInfo) {
      off += 4;
      bool ok = true;
      for (uint32_t i = 0; i < count; ++i) {
        if (off + 4 > n) { ok = false; break; }
        uint32_t ilen = net::GetU32(p + off); off += 4;
        if (off + ilen > n) { ok = false; break; }
        (*out)[i].info.assign(p + off, ilen); off += ilen;
      }
      if (!ok) break;
    } else if (tag == kMemberExtStats) {
      off += 4;
      if (off + 1 > n) break;
      const uint32_t nf = static_cast<uint8_t>(p[off]); off += 1;
      const uint32_t known = nf < kMemberStatsFields ? nf : kMemberStatsFields;
      bool ok = true;
      for (uint32_t i = 0; i < count; ++i) {
        if (off + 1 > n) { ok = false; break; }
        const bool has = p[off] != 0; off += 1;
        if (!has) continue;
        if (off + static_cast<size_t>(nf) * 8 > n) { ok = false; break; }
        uint64_t f[kMemberStatsFields] = {0};
        for (uint32_t k = 0; k < known; ++k) f[k] = net::GetU64(p + off + k * 8);
        off += static_cast<size_t>(nf) * 8;  // skip unknown appended fields too
        MemberStats& st = (*out)[i].stats;
        st.capacity_bytes = f[0]; st.used_bytes = f[1]; st.objects = f[2];
        st.hits_total = f[3]; st.misses_total = f[4]; st.evictions_total = f[5];
        st.puts_total = f[6]; st.uptime_seconds = f[7]; st.put_busy_total = f[8];
        st.dio_write_fallbacks = f[9]; st.ram_used_bytes = f[10]; st.ram_hits_total = f[11];
        (*out)[i].has_stats = true;
      }
      if (!ok) break;
    } else if (tag == kMemberExtHealth) {
      off += 4;
      bool ok = true;
      for (uint32_t i = 0; i < count; ++i) {
        if (off + 1 > n) { ok = false; break; }
        const bool has = p[off++] != 0;
        if (!has) continue;
        if (off + 5 > n) { ok = false; break; }
        MemberHealth health;
        health.ring_eligible = p[off++] != 0;
        const uint32_t devices = net::GetU32(p + off);
        off += 4;
        if (devices > 64) { ok = false; break; }
        health.ib_devices.reserve(devices);
        for (uint32_t j = 0; j < devices; ++j) {
          if (off + 4 > n) { ok = false; break; }
          const uint32_t name_len = net::GetU32(p + off);
          off += 4;
          if (name_len > 128 || off + name_len + 3 > n) {
            ok = false;
            break;
          }
          IbDeviceHealth device;
          device.name.assign(p + off, name_len);
          off += name_len;
          device.port_state = static_cast<uint8_t>(p[off++]);
          device.phys_state = static_cast<uint8_t>(p[off++]);
          device.query_ok = p[off++] != 0;
          health.ib_devices.push_back(std::move(device));
        }
        if (!ok) break;
        (*out)[i].health = std::move(health);
        (*out)[i].has_health = true;
      }
      if (!ok) break;
    } else {
      break;  // unknown/future tag: ignore the rest (forward compatible)
    }
  }
  return true;
}

// Content epoch for a membership view: an order-independent 64-bit hash of the
// member SET. Clients compare it to decide whether to rebuild their ring. The
// MDS uses this instead of etcd's global revision, which bumps on ANY cluster
// write (other groups too) and would trigger needless full-ring rebuilds. Sorting
// by id makes the hash canonical regardless of etcd's return order; hashing every
// field means a content-only change (e.g. an ip/weight edit) still bumps the epoch.
inline uint64_t MembersEpoch(std::vector<MemberInfo> ms) {
  std::sort(ms.begin(), ms.end(), [](const MemberInfo& a, const MemberInfo& b) {
    return a.id < b.id;
  });
  uint64_t h = 1469598103934665603ull;  // FNV-1a 64-bit offset basis
  auto mix = [&h](const void* p, size_t n) {
    const unsigned char* b = static_cast<const unsigned char*>(p);
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
  };
  for (const auto& m : ms) {
    uint32_t idn = static_cast<uint32_t>(m.id.size());
    uint32_t ipn = static_cast<uint32_t>(m.ip.size());
    mix(&idn, 4); mix(m.id.data(), m.id.size());  // length-prefixed: no field-boundary ambiguity
    mix(&ipn, 4); mix(m.ip.data(), m.ip.size());
    mix(&m.port, sizeof(m.port));
    mix(&m.weight, sizeof(m.weight));
    const bool eligible = m.RingEligible();
    mix(&eligible, sizeof(eligible));
  }
  return h;
}

}  // namespace dfkv

#endif  // DFKV_MEMBERSHIP_H_
