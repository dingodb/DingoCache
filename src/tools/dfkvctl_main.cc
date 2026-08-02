/* dfkvctl — CLI for a dfkv cluster.
 *   dfkvctl --members "n=ip:port,..." put   <key> <value>
 *   dfkvctl --members "n=ip:port,..." get   <key>
 *   dfkvctl --members "n=ip:port,..." exist <key>
 *   dfkvctl stat <node-ip:port>           # fetch a node's Prometheus metrics
 *   dfkvctl ring --mds <ep,...> --group <g>          # show the cluster ring + vnode share
 *   dfkvctl clients --mds <ep,...> --group <g>        # list cache consumers (inference instances)
 *   dfkvctl stat --all --mds <ep,...> --group <g> [--stat-port <p>]  # per-node metrics + aggregate
 *     The MDS now hands back each node's TCP stat port (MemberInfo.tcp_port), so stat --all
 *     reaches nodes automatically even in RDMA deploys. --stat-port <p> stays as a manual
 *     override (e.g. for older servers that predate tcp_port registration).
 * Native data commands require --namespace <model/raw-layout schema identity>. */
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "utils/con_hash.h"
#include "client/kv_client.h"
#include "mds/mds_member_poller.h"
#include "common/membership.h"
#include "utils/prom_parse.h"
#include "transport/tcp_transport.h"
#include "common/version.h"
#include "utils/wire_limits.h"

using namespace dfkv;  // NOLINT

static std::vector<std::string> SplitCsv(const std::string& s) {
  std::vector<std::string> out;
  for (size_t i = 0, j; i <= s.size(); i = j + 1) {
    j = s.find(',', i);
    if (j == std::string::npos) j = s.size();
    if (j > i) out.push_back(s.substr(i, j - i));
  }
  return out;
}

// One-shot MDS member query via the poller (PollOnce fires the callback on the
// first epoch). Returns true if the MDS answered.
static bool QueryMembers(const std::string& mds, const std::string& group,
                         std::vector<MemberInfo>* out) {
  auto eps = SplitCsv(mds);
  if (eps.empty()) return false;
  bool got = false;
  MdsMemberPoller poller(eps, group,
                         [&](const std::vector<MemberInfo>& ms) { *out = ms; got = true; },
                         1000, 2000);
  // Try up to one full round of the endpoint list: PollOnce picks a single MDS
  // per call and MarkFailed's backoff steers the next Pick to a different one,
  // so a dead first endpoint no longer fails the whole command while other MDS
  // instances are healthy (RUNBOOK uses `dfkvctl ring` to verify upgrades).
  for (size_t i = 0; i < eps.size() && !got; ++i) poller.PollOnce();
  return got;
}

// One-shot MDS client-list query via the poller (PollOnce fires the callback on
// the first epoch). Returns true if the MDS answered. Mirrors QueryMembers but
// carries kListClients so the MDS reads /clients/<id> instead of /members/<id>.
static bool QueryClients(const std::string& mds, const std::string& group,
                         std::vector<MemberInfo>* out) {
  auto eps = SplitCsv(mds);
  if (eps.empty()) return false;
  bool got = false;
  // Reuse the member poller's endpoint failover by issuing a single kListClients
  // roundtrip per endpoint — the poller's Pick/MarkFailed steers the next try.
  for (const auto& ep : eps) {
    int fd = net::Dial(ep, 2000, 2000);
    if (fd < 0) continue;
    char pre[kReqPrefix];
    EncodeReq(pre, WireOp::kListClients, BlockKey{}, 0, 0, group.size());
    // An empty group is a valid query (server-side default group): it must not
    // short-circuit the send into a fake failure ('&&' bug — poller uses '||').
    bool ok = net::WriteAll(fd, pre, kReqPrefix) &&
              (group.empty() || net::WriteAll(fd, group.data(), group.size()));
    std::string data;
    if (ok) {
      char rp[kRespPrefix];
      Status st = Status::kInvalid;
      uint64_t dlen = 0;
      ok = net::ReadAll(fd, rp, kRespPrefix) &&
           DecodeResp(rp, &st, &dlen, wire_limits::kMdsMaxRespData) && st == Status::kOk;
      if (ok) { data.resize(dlen); ok = (dlen == 0) || net::ReadAll(fd, &data[0], dlen); }
    }
    ::close(fd);
    if (!ok) continue;
    uint64_t epoch = 0;
    if (DecodeMembers(data.data(), data.size(), out, &epoch)) { got = true; break; }
  }
  return got;
}

// Extract the value of `key=` from a "k=v,k=v" info string, or "" if absent.
static std::string InfoField(const std::string& info, const std::string& key) {
  std::string tok = key + "=";
  size_t p = info.find(tok);
  if (p == std::string::npos) return "";
  p += tok.size();
  size_t e = info.find(',', p);
  return info.substr(p, (e == std::string::npos ? info.size() : e) - p);
}

// Raw kListGroups roundtrip against any one reachable MDS endpoint: returns
// the newline-joined distinct group names.
static bool QueryGroups(const std::string& mds, std::vector<std::string>* out) {
  for (const auto& ep : SplitCsv(mds)) {
    int fd = net::Dial(ep, 2000, 2000);
    if (fd < 0) continue;
    char pre[kReqPrefix];
    EncodeReq(pre, WireOp::kListGroups, BlockKey{}, 0, 0, 0);
    bool ok = net::WriteAll(fd, pre, kReqPrefix);
    std::string data;
    if (ok) {
      char rp[kRespPrefix];
      Status st = Status::kInvalid;
      uint64_t dlen = 0;
      ok = net::ReadAll(fd, rp, kRespPrefix) &&
           DecodeResp(rp, &st, &dlen, wire_limits::kMdsMaxRespData) && st == Status::kOk;
      if (ok) { data.resize(dlen); ok = (dlen == 0) || net::ReadAll(fd, &data[0], dlen); }
    }
    ::close(fd);
    if (!ok) continue;
    out->clear();
    size_t i = 0;
    while (i <= data.size()) {
      size_t nl = data.find('\n', i);
      if (nl == std::string::npos) nl = data.size();
      if (nl > i) out->push_back(data.substr(i, nl - i));
      if (nl == data.size()) break;
      i = nl + 1;
    }
    return true;
  }
  return false;
}

static std::string HumanBytes(uint64_t b) {
  const char* u[] = {"B", "K", "M", "G", "T", "P"};
  double v = static_cast<double>(b);
  int i = 0;
  while (v >= 1024.0 && i < 5) { v /= 1024.0; ++i; }
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.1f%s", v, u[i]);
  return buf;
}

static std::string HumanSecs(uint64_t s) {
  char buf[32];
  if (s >= 86400) std::snprintf(buf, sizeof(buf), "%llud%lluh", (unsigned long long)s / 86400, ((unsigned long long)s % 86400) / 3600);
  else if (s >= 3600) std::snprintf(buf, sizeof(buf), "%lluh%llum", (unsigned long long)s / 3600, ((unsigned long long)s % 3600) / 60);
  else std::snprintf(buf, sizeof(buf), "%llum", (unsigned long long)s / 60);
  return buf;
}

// Per-ring runtime stats straight from the MDS (values ride each member's
// heartbeat, ~10s freshness): the fleet answer without touching any node.
// dfkvctl stat --all remains the per-node deep-dive (full /metrics).
static int CmdStats(const std::string& mds, const std::string& group) {
  if (mds.empty()) { std::fprintf(stderr, "stats needs --mds ip:port[,...]\n"); return 2; }
  std::vector<MemberInfo> ms;
  if (!QueryMembers(mds, group, &ms)) { std::fprintf(stderr, "stats: MDS query failed\n"); return 1; }
  std::printf("group=%s members=%zu (stats as of each node's last heartbeat)\n", group.c_str(), ms.size());
  std::printf("%-16s %6s %14s %5s %8s %6s %7s %6s %7s %8s %7s\n",
              "ID", "WEIGHT", "USED/CAP", "UTIL", "OBJ", "HIT%", "EVIC", "BUSY", "DIO-FB", "RAM-USED", "UPTIME");
  MemberStats sum;
  uint64_t missing = 0;
  for (const auto& m : ms) {
    if (!m.has_stats) {
      std::printf("%-16s %6u %14s\n", m.id.c_str(), m.weight, "(no stats: pre-upgrade node)");
      ++missing;
      continue;
    }
    const MemberStats& st = m.stats;
    const double util = st.capacity_bytes
        ? 100.0 * static_cast<double>(st.used_bytes) / static_cast<double>(st.capacity_bytes) : 0.0;
    const uint64_t lookups = st.hits_total + st.misses_total;
    const double hitp = lookups ? 100.0 * static_cast<double>(st.hits_total) / static_cast<double>(lookups) : 0.0;
    std::printf("%-16s %6u %14s %4.0f%% %8llu %5.1f%% %7llu %6llu %7llu %8s %7s\n",
                m.id.c_str(), m.weight,
                (HumanBytes(st.used_bytes) + "/" + HumanBytes(st.capacity_bytes)).c_str(), util,
                (unsigned long long)st.objects, hitp,
                (unsigned long long)st.evictions_total, (unsigned long long)st.put_busy_total,
                (unsigned long long)st.dio_write_fallbacks,
                st.ram_used_bytes ? HumanBytes(st.ram_used_bytes).c_str() : "-",
                HumanSecs(st.uptime_seconds).c_str());
    sum.capacity_bytes += st.capacity_bytes; sum.used_bytes += st.used_bytes;
    sum.objects += st.objects; sum.hits_total += st.hits_total; sum.misses_total += st.misses_total;
    sum.evictions_total += st.evictions_total; sum.put_busy_total += st.put_busy_total;
    sum.dio_write_fallbacks += st.dio_write_fallbacks; sum.ram_used_bytes += st.ram_used_bytes;
  }
  const uint64_t lk = sum.hits_total + sum.misses_total;
  // Surface registered consumers on the same line so `dfkvctl stats` answers
  // "and who is using this ring" without a second command.
  std::vector<MemberInfo> cs;
  size_t nclients = QueryClients(mds, group, &cs) ? cs.size() : 0;
  std::printf("-- group=%s: %zu nodes  %s/%s (%.0f%%)  hit %.1f%%  evic %llu  busy %llu  dio-fb %llu  ram %s  stats-missing %llu  clients %zu\n",
              group.c_str(), ms.size(), HumanBytes(sum.used_bytes).c_str(), HumanBytes(sum.capacity_bytes).c_str(),
              sum.capacity_bytes ? 100.0 * (double)sum.used_bytes / (double)sum.capacity_bytes : 0.0,
              lk ? 100.0 * (double)sum.hits_total / (double)lk : 0.0,
              (unsigned long long)sum.evictions_total, (unsigned long long)sum.put_busy_total,
              (unsigned long long)sum.dio_write_fallbacks, HumanBytes(sum.ram_used_bytes).c_str(),
              (unsigned long long)missing, nclients);
  return 0;
}

static int CmdStatsAllGroups(const std::string& mds) {
  std::vector<std::string> groups;
  if (!QueryGroups(mds, &groups)) { std::fprintf(stderr, "stats --all: kListGroups failed (MDS < 1.10?)\n"); return 1; }
  if (groups.empty()) { std::printf("no groups registered\n"); return 0; }
  int rc = 0;
  for (size_t i = 0; i < groups.size(); ++i) {
    if (i) std::printf("\n");
    rc |= CmdStats(mds, groups[i]);
  }
  return rc;
}

static int CmdRing(const std::string& mds, const std::string& group) {
  if (mds.empty()) { std::fprintf(stderr, "ring needs --mds ip:port[,...]\n"); return 2; }
  std::vector<MemberInfo> ms;
  if (!QueryMembers(mds, group, &ms)) { std::fprintf(stderr, "ring: MDS query failed\n"); return 1; }
  ConHash ring;
  for (const auto& m : ms) ring.AddNode(m.id, m.weight);
  ring.Build();
  auto pc = ring.NodePointCounts();
  size_t total = ring.RingSize();
  std::printf("group=%s members=%zu ring_points=%zu\n", group.c_str(), ms.size(), total);
  std::printf("%-16s %-22s %6s %8s %7s  %s\n", "ID", "ADDR", "WEIGHT", "VNODES", "SHARE", "INFO");
  for (const auto& m : ms) {
    std::string addr = m.ip + ":" + std::to_string(m.port);
    size_t v = pc.count(m.id) ? pc[m.id] : 0;
    double share = total ? 100.0 * static_cast<double>(v) / static_cast<double>(total) : 0.0;
    // INFO = the node's self-description from register/heartbeat (version, engine,
    // cap, ...). "-" = node predates info reporting (itself a version signal).
    std::printf("%-16s %-22s %6u %8zu %6.1f%%  %s\n", m.id.c_str(), addr.c_str(), m.weight,
                v, share, m.info.empty() ? "-" : m.info.c_str());
  }
  return 0;
}

static int CmdClients(const std::string& mds, const std::string& group) {
  if (mds.empty()) { std::fprintf(stderr, "clients needs --mds ip:port[,...]\n"); return 2; }
  std::vector<MemberInfo> cs;
  if (!QueryClients(mds, group, &cs)) { std::fprintf(stderr, "clients: MDS query failed\n"); return 1; }
  std::printf("group=%s clients=%zu (only upgraded clients register)\n", group.c_str(), cs.size());
  if (cs.empty()) return 0;
  std::printf("%-28s %-10s %-24s %-14s %-8s %s\n", "ID", "TYPE", "MODEL", "ROLE", "TP", "INFO");
  for (const auto& m : cs) {
    std::string type = InfoField(m.info, "type");
    std::string model = InfoField(m.info, "model");
    std::string role = InfoField(m.info, "role");
    std::string tp = InfoField(m.info, "tp_size");
    std::printf("%-28s %-10s %-24s %-14s %-8s %s\n", m.id.c_str(),
                type.empty() ? "-" : type.c_str(),
                model.empty() ? "-" : model.c_str(),
                role.empty() ? "-" : role.c_str(),
                tp.empty() ? "-" : tp.c_str(),
                m.info.empty() ? "-" : m.info.c_str());
  }
  return 0;
}

static int CmdStatAll(const std::string& mds, const std::string& group,
                      uint32_t stat_port) {
  if (mds.empty()) { std::fprintf(stderr, "stat --all needs --mds ip:port[,...]\n"); return 2; }
  std::vector<MemberInfo> ms;
  if (!QueryMembers(mds, group, &ms)) { std::fprintf(stderr, "stat --all: MDS query failed\n"); return 1; }
  TcpTransport t; t.set_timeouts(2000, 3000);
  uint64_t T_used = 0, T_obj = 0, T_hit = 0, T_miss = 0, T_w = 0, T_r = 0;
  std::printf("%-14s %-20s %10s %9s %9s %9s %6s\n",
              "ID", "ADDR", "USED_MB", "OBJECTS", "HITS", "MISSES", "HIT%");
  for (const auto& m : ms) {
    // Port to reach kStats on: an explicit --stat-port wins; else the server-registered
    // TCP wire/stat port from the MDS (m.tcp_port); else fall back to the member's data
    // port (the rdma-port in an RDMA deploy, where it would print (unreachable)).
    uint32_t sp = stat_port ? stat_port : (m.tcp_port ? m.tcp_port : m.port);
    std::string addr = m.ip + ":" + std::to_string(sp), text;
    if (t.Stats(addr, &text) != Status::kOk) {
      std::printf("%-14s %-20s   (unreachable)\n", m.id.c_str(), addr.c_str());
      continue;
    }
    uint64_t used = PromMetricValue(text, "dfkv_used_bytes");
    uint64_t obj = PromMetricValue(text, "dfkv_objects");
    uint64_t hit = PromMetricValue(text, "dfkv_cache_hit_total");
    uint64_t miss = PromMetricValue(text, "dfkv_cache_miss_total");
    uint64_t w = PromMetricValue(text, "dfkv_bytes_written_total");
    uint64_t r = PromMetricValue(text, "dfkv_bytes_read_total");
    double hr = (hit + miss) ? 100.0 * static_cast<double>(hit) / static_cast<double>(hit + miss) : 0.0;
    std::printf("%-14s %-20s %10.1f %9llu %9llu %9llu %5.1f%%\n", m.id.c_str(), addr.c_str(),
                used / 1048576.0, (unsigned long long)obj, (unsigned long long)hit,
                (unsigned long long)miss, hr);
    T_used += used; T_obj += obj; T_hit += hit; T_miss += miss; T_w += w; T_r += r;
  }
  double thr = (T_hit + T_miss) ? 100.0 * static_cast<double>(T_hit) / static_cast<double>(T_hit + T_miss) : 0.0;
  std::printf("TOTAL used=%.1fMB objects=%llu hits=%llu misses=%llu hit%%=%.1f bytes_w=%llu bytes_r=%llu\n",
              T_used / 1048576.0, (unsigned long long)T_obj, (unsigned long long)T_hit,
              (unsigned long long)T_miss, thr, (unsigned long long)T_w, (unsigned long long)T_r);
  return 0;
}

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
  if (WantsVersion(argc, argv)) { std::printf("dfkvctl %s\n", Version()); return 0; }
  std::string members;
  std::string key_namespace;
  std::string mds, group = "default";
  bool all = false;
  uint32_t stat_port = 0;  // override; 0 = use MDS-provided tcp_port, else the member's port
  std::vector<std::string> pos;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    // strtoull/strtoul (not stoull/stoul): the std:: variants THROW on a
    // non-numeric arg, terminating the tool; strto* return 0 instead.
    auto nv32 = [&](uint32_t* d) { if (i + 1 < argc) *d = (uint32_t)std::strtoul(argv[++i], nullptr, 0); };
    if (a == "--members" && i + 1 < argc) members = argv[++i];
    else if (a == "--mds" && i + 1 < argc) mds = argv[++i];
    else if (a == "--group" && i + 1 < argc) group = argv[++i];
    else if (a == "--all") all = true;
    else if (a == "--stat-port") nv32(&stat_port);
    else if (a == "--namespace" && i + 1 < argc) key_namespace = argv[++i];
    else pos.push_back(a);
  }
  if (pos.empty()) { std::fprintf(stderr, "usage: dfkvctl [--members ...] [--namespace ...] put|get|exist|stat|stats|ring ...\n"); return 2; }
  const std::string& cmd = pos[0];

  if (cmd == "ring") return CmdRing(mds, group);
  if (cmd == "clients") return CmdClients(mds, group);

  if (cmd == "stats") {
    if (all) return CmdStatsAllGroups(mds);
    return CmdStats(mds, group);
  }
  if (cmd == "stat") {
    if (all) return CmdStatAll(mds, group, stat_port);
    if (pos.size() < 2) { std::fprintf(stderr, "stat <node-ip:port>  |  stat --all --mds ... [--stat-port <p>]\n"); return 2; }
    TcpTransport t; std::string text;
    if (t.Stats(pos[1], &text) != Status::kOk) { std::fprintf(stderr, "stat failed\n"); return 1; }
    std::fputs(text.c_str(), stdout);
    return 0;
  }

  auto mem = ParseMembers(members);
  if (mem.empty()) { std::fprintf(stderr, "need --members name=ip:port,...\n"); return 2; }
  if (key_namespace.empty()) { std::fprintf(stderr, "need --namespace model/raw-layout-schema\n"); return 2; }
  KVClient c(mem, key_namespace);

  if (cmd == "put" && pos.size() >= 3) {
    bool ok = c.Put(pos[1], pos[2].data(), pos[2].size());
    std::printf("%s\n", ok ? "OK" : "FAIL"); return ok ? 0 : 1;
  }
  if (cmd == "get" && pos.size() >= 2) {
    std::string out;
    if (!c.GetAuto(pos[1], &out)) { std::printf("(miss)\n"); return 1; }
    std::fwrite(out.data(), 1, out.size(), stdout); std::printf("\n"); return 0;
  }
  if (cmd == "exist" && pos.size() >= 2) {
    bool e = c.Exist(pos[1]); std::printf("%s\n", e ? "true" : "false"); return e ? 0 : 1;
  }
  std::fprintf(stderr, "unknown/!args: %s\n", cmd.c_str());
  return 2;
}
