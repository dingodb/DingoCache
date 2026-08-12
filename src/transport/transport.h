/* Transport abstraction between the KV client and cache nodes.
 * Real build: a brpc-backed impl over dingofs RemoteBlockCache.
 * Test/harness build: TcpTransport (POSIX sockets). */
#ifndef DFKV_TRANSPORT_H_
#define DFKV_TRANSPORT_H_

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "common/status.h"
#include "common/kv_types.h"
#include "transport/wire.h"       // WireOp, kReqPrefix, kRespPrefix, Encode/DecodeReq/Resp

namespace dfkv {
inline bool ValidBuffer(const void* data, size_t size) {
  return size == 0 || data != nullptr;
}

inline bool CheckedAddSize(size_t value, size_t add, size_t* out) {
  if (add > std::numeric_limits<size_t>::max() - value) return false;
  *out = value + add;
  return true;
}

inline std::vector<Status> InvalidStatuses(size_t count) {
  return std::vector<Status>(count, Status::kInvalid);
}


// One write in a batch (data must outlive the CacheMany call).
struct CacheItem { BlockKey key; const void* data; size_t len; };
// One zero-copy read target: up to `n` bytes are written into `payload`.
// The authoritative stored length is returned separately by the read call.
struct RangeDst { void* payload; size_t n; };
// One zero-copy write source. The raw value is gathered directly from payload.
// The buffer must outlive CacheFrom and remain unmodified until it returns.
struct CacheSrc {
  BlockKey key;
  const void* payload;
  size_t payload_len;
};

// Scatter-gather write source: one raw value formed by concatenating N caller
// buffers in order. RDMA gathers them without a client-side concat copy.
struct CacheSrcMulti {
  BlockKey key;
  std::vector<std::pair<const void*, size_t>> payloads;
};
// Scatter-gather read target: the stored payload (one concatenated blob) is
// scattered across N caller buffers in order, each receiving its own size worth
// of bytes (RDMA multi-SGE RECV). The destination capacity = sum of segment sizes.
struct RangeDstMulti {
  std::vector<std::pair<void*, size_t>> payloads;  // (ptr, size) in order
};
// Common per-endpoint connection-pool policy. A transport may ignore it when
// its connection lifecycle is protocol-defined, but pooled transports use the
// same bounded defaults so client fan-out cannot create an unbounded number of
// sockets/QPs. Connections are always scoped to one routed endpoint: selecting
// a connection must never change the cache node chosen for a key.
struct EndpointPoolOptions {
  size_t min_connections = 1;
  size_t max_connections = 8;
  uint64_t idle_timeout_ms = 30000;
  uint64_t backoff_base_ms = 50;
  uint64_t backoff_max_ms = 5000;
  uint64_t acquire_timeout_ms = 10000;
};

// One peer rail advertised by the membership topology. Device names are the
// shared placement binding between hosts; health is the peer's runtime view.
struct PeerRailTopology {
  std::string name;
  bool healthy = false;
};

// Complete peer-specific topology update. generation is the canonical
// membership topology epoch, not a transport-local counter. Transports must
// treat snapshots as immutable so one logical operation cannot mix epochs.
struct PeerTopology {
  std::string peer_addr;
  uint64_t generation = 0;
  bool complete = false;
  std::vector<PeerRailTopology> rails;
};


class Transport {
 public:
  virtual ~Transport() = default;
  // Synchronous, durable-visible write (server uses KVStore::Cache).
  virtual Status Cache(const std::string& node, const BlockKey& key,
                       const void* data, size_t len) = 0;
  virtual Status Range(const std::string& node, const BlockKey& key,
                       uint64_t offset, uint64_t length, std::string* out,
                       uint64_t* value_len = nullptr) = 0;
  virtual Status Lookup(const std::string& node, const BlockKey& key,
                        uint64_t* value_len) = 0;
  virtual Status Exist(const std::string& node, const BlockKey& key,
                       bool* exist) = 0;

  // Explicitly drop a block on its owning node (LMCache L2 eviction). Default:
  // not supported; TcpTransport/RdmaTransport override it with a kRemove round
  // trip. kOk = removed, kNotFound = absent, kIOError = transport failure.
  virtual Status Remove(const std::string& node, const BlockKey& key) {
    (void)node; (void)key; return Status::kInvalid;
  }

  // Query a node's advertised cluster member list (for discovery). Default: not
  // supported; TCP/RDMA override it. *out = "name=ip:port,name=ip:port,...".
  virtual Status Members(const std::string& node, std::string* out) {
    (void)node; (void)out; return Status::kInvalid;
  }

  // Batch remove for one node. Default = sequential loop over Remove (eviction
  // is off the hot path, so neither transport bothers to pipeline it). Returns
  // the per-key Status for caller-side health accounting.
  virtual std::vector<Status> RemoveMany(const std::string& node,
                                         const std::vector<BlockKey>& keys) {
    std::vector<Status> r;
    r.reserve(keys.size());
    for (const auto& k : keys) r.push_back(Remove(node, k));
    return r;
  }

  // Transport-level Prometheus metrics (e.g. pooled connection load/lifecycle,
  // RDMA per-rail state, MR registrations). Default empty; concrete transports
  // override it. Folded into the client snapshot, rendered off the datapath.
  virtual std::string MetricsText() const { return ""; }

  // Register a caller-owned stable memory region (for example the whole host KV
  // pool) for zero-copy transfer. RDMA registers it once per shared device PD;
  // buffers inside resolve by range lookup without per-op registration. Calling
  // again with the same base and a larger size grows the declaration. Returns
  // true only when the requested range is ready for use; default copying
  // transports such as TCP need no registration and always succeed.
  virtual bool RegisterMemory(void* base, size_t size) {
    (void)base;
    (void)size;
    return true;
  }

  // Ring-scale hint from the client's adopted MDS membership. Connection-
  // oriented transports size process-wide resource budgets from it (demand
  // grows with nodes x rails, not with any constant), so a fixed default can
  // never starve a large ring into admission timeouts (.issue/0812-004).
  // Called on every membership adoption; implementations must tolerate
  // repeated and concurrent calls. Default: stateless transports ignore it.
  virtual void OnTopologyHint(size_t nodes) { (void)nodes; }

  // Peer-aware rail topology is independent of the scalar ring-size budget
  // hint. Stateful transports publish an immutable per-address snapshot;
  // transports without heterogeneous path selection ignore it.
  virtual void OnPeerTopology(const PeerTopology& topology) {
    (void)topology;
  }

  // True if CacheMany/RangeMany pipeline requests on one connection (RDMA). When
  // false (TCP), the client parallelizes batches across items with its own
  // threads instead, since the per-node loop here would be sequential.
  virtual bool pipelined() const { return false; }

  // Max payload segments one scatter-gather key may carry end-to-end. On RDMA
  // this is the live negotiated max_sge minus the header SGE; the TCP default
  // keeps the documented ConnectX-era 29 so batches built against it stay
  // valid everywhere. Callers (connectors, via the C ABI) size their SG
  // grouping from this instead of hard-coding 29.
  virtual size_t MaxSgPayloadSegs() const { return 29; }

  // Batch variants for one node. Default = sequential loop; RDMA overrides these
  // to pipeline multiple requests in flight on a single connection. All keys in
  // a RangeMany share (offset, length).
  virtual std::vector<Status> CacheMany(const std::string& node,
                                        const std::vector<CacheItem>& items) {
    std::vector<Status> r;
    r.reserve(items.size());
    for (const auto& it : items) {
      r.push_back(it.len != 0 && ValidBuffer(it.data, it.len)
                      ? Cache(node, it.key, it.data, it.len)
                      : Status::kInvalid);
    }
    return r;
  }
  virtual std::vector<Status> RangeMany(
      const std::string& node, const std::vector<BlockKey>& keys,
      uint64_t offset, uint64_t length, std::vector<std::string>* outs,
      std::vector<uint64_t>* value_lens = nullptr) {
    if (outs == nullptr) {
      if (value_lens) value_lens->assign(keys.size(), 0);
      return InvalidStatuses(keys.size());
    }
    outs->assign(keys.size(), std::string());
    if (value_lens) value_lens->assign(keys.size(), 0);
    std::vector<Status> r;
    r.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i)
      r.push_back(Range(node, keys[i], offset, length, &(*outs)[i],
                        value_lens ? &(*value_lens)[i] : nullptr));
    return r;
  }

  // Batch existence probe for one node. Default = sequential loop; RDMA overrides
  // it to pipeline kExist requests on a single connection (one Acquire/node)
  // instead of one round trip per key. (*exists)[i] is 1 iff keys[i] is present;
  // the returned Status[i] is the raw per-key outcome for caller-side health
  // accounting (kIOError => transport failure; anything else => node responded).
  virtual std::vector<Status> ExistMany(const std::string& node,
                                        const std::vector<BlockKey>& keys,
                                        std::vector<char>* exists) {
    if (exists == nullptr) return InvalidStatuses(keys.size());
    exists->assign(keys.size(), 0);
    std::vector<Status> r;
    r.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      bool e = false;
      Status st = Exist(node, keys[i], &e);
      (*exists)[i] = (st == Status::kOk && e) ? 1 : 0;
      r.push_back(st);
    }
    return r;
  }

  // Zero-copy read: up to dsts[i].n raw stored bytes land directly in
  // dsts[i].payload. value_lens[i] is the authoritative full stored size.
  // Default: Range + copy; RDMA overrides with direct writes.
  virtual std::vector<Status> RangeInto(
      const std::string& node, const std::vector<BlockKey>& keys,
      const std::vector<RangeDst>& dsts,
      std::vector<uint64_t>* value_lens) {
    if (value_lens) value_lens->assign(keys.size(), 0);
    if (dsts.size() != keys.size()) return InvalidStatuses(keys.size());
    std::vector<Status> r(keys.size(), Status::kInvalid);
    for (size_t i = 0; i < keys.size(); ++i) {
      if (!ValidBuffer(dsts[i].payload, dsts[i].n)) continue;
      std::string raw;
      uint64_t stored = 0;
      Status st = Range(node, keys[i], 0, dsts[i].n, &raw, &stored);
      r[i] = st;
      if (value_lens) (*value_lens)[i] = stored;
      if (st != Status::kOk) continue;
      size_t n = std::min(raw.size(), dsts[i].n);
      if (n) std::memcpy(dsts[i].payload, raw.data(), n);
    }
    return r;
  }

  // Zero-copy batched write. Default transports route caller buffers directly
  // through CacheMany; RDMA overrides with registered scatter sends.
  virtual std::vector<Status> CacheFrom(const std::string& node,
                                        const std::vector<CacheSrc>& srcs) {
    std::vector<CacheItem> items;
    items.reserve(srcs.size());
    for (const auto& src : srcs)
      items.push_back(CacheItem{src.key, src.payload, src.payload_len});
    return CacheMany(node, items);
  }

  // Scatter-gather write: each key's payload is the concatenation of N caller
  // segments. Default: concatenate the segments into a single CacheSrc and route
  // through CacheFrom (no payload zero-copy, but correct on any transport). The
  // RDMA transport overrides this to gather the segments via a multi-SGE SEND.
  virtual std::vector<Status> CacheFromMulti(
      const std::string& node, const std::vector<CacheSrcMulti>& srcs) {
    std::vector<Status> result(srcs.size(), Status::kInvalid);
    std::vector<std::vector<char>> bufs(srcs.size());
    std::vector<CacheSrc> flat;
    std::vector<size_t> indices;
    flat.reserve(srcs.size());
    indices.reserve(srcs.size());
    for (size_t i = 0; i < srcs.size(); ++i) {
      size_t total = 0;
      bool valid = true;
      for (const auto& p : srcs[i].payloads) {
        size_t next = 0;
        if (!ValidBuffer(p.first, p.second) ||
            !CheckedAddSize(total, p.second, &next)) {
          valid = false;
          break;
        }
        total = next;
      }
      if (!valid || total == 0) continue;
      bufs[i].resize(total);
      size_t off = 0;
      for (const auto& p : srcs[i].payloads) {
        if (p.second) std::memcpy(bufs[i].data() + off, p.first, p.second);
        off += p.second;
      }
      indices.push_back(i);
      flat.push_back(CacheSrc{srcs[i].key,
                              total ? bufs[i].data() : nullptr, total});
    }
    const auto valid_result = CacheFrom(node, flat);
    for (size_t i = 0; i < indices.size() && i < valid_result.size(); ++i)
      result[indices[i]] = valid_result[i];
    return result;
  }

  // Scatter-gather read: raw stored bytes are split across each key's caller
  // buffers in order. The default implementation reads once into a contiguous
  // scratch value and split-copies it, so every transport is correct without
  // scatter support. RDMA overrides this with a multi-SGE receive directly into
  // registered destinations. out_lens[i] is the authoritative stored length
  // (zero on miss), independent of destination capacity.
  virtual std::vector<Status> RangeIntoMulti(
      const std::string& node, const std::vector<BlockKey>& keys,
      const std::vector<RangeDstMulti>& dsts,
      std::vector<size_t>* out_lens) {
    const size_t n = keys.size();
    if (out_lens) out_lens->assign(n, 0);
    if (n == 0) return {};
    if (dsts.size() != n) return InvalidStatuses(n);
    std::vector<Status> result(n, Status::kInvalid);
    std::vector<std::vector<char>> scratch(n);
    std::vector<BlockKey> valid_keys;
    std::vector<RangeDst> flat;
    std::vector<size_t> indices;
    valid_keys.reserve(n);
    flat.reserve(n);
    indices.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      size_t cap = 0;
      bool valid = true;
      for (const auto& p : dsts[i].payloads) {
        size_t next = 0;
        if (!ValidBuffer(p.first, p.second) ||
            !CheckedAddSize(cap, p.second, &next)) {
          valid = false;
          break;
        }
        cap = next;
      }
      if (!valid) continue;
      scratch[i].resize(cap);
      indices.push_back(i);
      valid_keys.push_back(keys[i]);
      flat.push_back(RangeDst{cap ? scratch[i].data() : nullptr, cap});
    }
    std::vector<uint64_t> stored;
    const auto valid_result = RangeInto(node, valid_keys, flat, &stored);
    for (size_t valid_i = 0;
         valid_i < indices.size() && valid_i < valid_result.size(); ++valid_i) {
      const size_t i = indices[valid_i];
      result[i] = valid_result[valid_i];
      if (result[i] != Status::kOk) continue;
      if (valid_i >= stored.size()) {
        result[i] = Status::kInvalid;
        continue;
      }
      if (stored[valid_i] > std::numeric_limits<size_t>::max() ||
          static_cast<size_t>(stored[valid_i]) > scratch[i].size()) {
        result[i] = Status::kInvalid;
        continue;
      }
      size_t remaining = static_cast<size_t>(stored[valid_i]);
      size_t off = 0;
      for (const auto& p : dsts[i].payloads) {
        const size_t copy = std::min(p.second, remaining);
        if (copy) std::memcpy(p.first, scratch[i].data() + off, copy);
        off += copy;
        remaining -= copy;
      }
      if (out_lens) (*out_lens)[i] = static_cast<size_t>(stored[valid_i]);
    }
    return result;
  }
};

}  // namespace dfkv

#endif  // DFKV_TRANSPORT_H_
