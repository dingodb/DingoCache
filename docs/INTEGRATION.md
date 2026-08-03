# Optional: fuse dfkv semantics into production dingo-cache (brpc + MDS)

> **Status — archived design note (pre-v2), read first.** The primary, supported
> way to run dfkv is the **standalone repo** with its own daemon + native-verbs
> RDMA transport (`dfkv_server`, see `docs/DEPLOY.md`). The standalone path
> includes its **own lightweight MDS** (`dfkv_mds` + etcd) for dynamic membership
> and service discovery — it does **not** require or touch the production
> dingo-cache MDS. This document is an **optional, aspirational** guide for an
> alternative path that was never productized: re-implementing the same
> Cache/Range/Exist semantics *inside* the production `dingo-cache` (brpc + MDS +
> DiskCache) so HiCache could talk to the existing cache fleet over brpc instead
> of dfkv_server. It was last reconciled against the **pre-v2 codebase** (the
> v1.x/two-sided-transport era) and does **not** describe the current v2
> one-sided datapath, `dfkv_open_v2` constructor, or canonical identity codec —
> re-confirm every line ref and mechanism claim on the current build tag before
> acting on it. It is **not required** for deployment. Most users want
> `DEPLOY.md`, not this.

To wire the semantics into the full dingofs build, apply the following.
**Compile in the full toolchain (gcc-13 / cmake-3.30 / dingo-eureka).**

## 1. proto (DONE on this branch)
`proto/dingofs/blockcache.proto`: added `ExistRequest/ExistResponse`,
`rpc Exist`, `rpc SyncCache(CacheRequest) returns (CacheResponse)`. Regenerate
`blockcache.pb.h`.

## 2. cachegroup service handlers — `src/cache/cachegroup/service.{h,cc}`
Add (mirror the existing `Range`/`Cache`/`Ping` handlers):
```cpp
// service.h (declarations)
void Exist(google::protobuf::RpcController*, const pb::cache::ExistRequest*,
           pb::cache::ExistResponse*, google::protobuf::Closure*) override;
void SyncCache(google::protobuf::RpcController*, const pb::cache::CacheRequest*,
               pb::cache::CacheResponse*, google::protobuf::Closure*) override;
```
```cpp
// service.cc
void BlockCacheServiceImpl::Exist(google::protobuf::RpcController*,
    const pb::cache::ExistRequest* req, pb::cache::ExistResponse* resp,
    google::protobuf::Closure* done) {
  brpc::ClosureGuard g(done);
  auto ctx = NewContext();
  bool exist = node_->Exist(ctx, FromContextPB(req->block_ctx()));
  resp->set_exist(exist);
  resp->set_status(ToPBErr(Status::OK()));
}
void BlockCacheServiceImpl::SyncCache(google::protobuf::RpcController* cntl_base,
    const pb::cache::CacheRequest* req, pb::cache::CacheResponse* resp,
    google::protobuf::Closure* done) {
  brpc::ClosureGuard g(done);
  auto* cntl = static_cast<brpc::Controller*>(cntl_base);
  auto ctx = NewContext();
  IOBuffer buffer(cntl->request_attachment().movable());
  auto st = CheckBodySize(req->block_size(), buffer.Size());
  if (st.ok()) st = node_->SyncCache(ctx, FromContextPB(req->block_ctx()),
                                     Block(std::move(buffer)));
  resp->set_status(ToPBErr(st));
}
```

## 3. CacheNode — `src/cache/cachegroup/node.{h,cc}`
```cpp
// node.h
bool   Exist(ContextSPtr, const BlockContext&);
Status SyncCache(ContextSPtr, const BlockContext&, const Block&);
```
```cpp
// node.cc
bool CacheNode::Exist(ContextSPtr, const BlockContext& bctx) {
  if (!IsRunning()) return false;
  return block_cache_->IsCached(bctx);            // local only, never RetrieveStorage
}
Status CacheNode::SyncCache(ContextSPtr ctx, const BlockContext& bctx, const Block& b) {
  if (!IsRunning()) return Status::CacheDown("down");
  return block_cache_->Cache(ctx, bctx, b);       // SYNC Cache: WriteFile+Add before return
}
```
Also `peer.cc NextTimeoutMs`/`ShouldRetry`: add `"Exist"` and `"SyncCache"`
method cases (else the brpc client hits `CHECK(false) "Unknown rpc method"`).
And in `upstream.{h,cc}` add `SendExistRequest`/`SendSyncCacheRequest`
(mirror `SendRangeRequest`/`SendCacheRequest`), and expose
`RemoteBlockCacheImpl::Exist`/`SyncCache`.

## 4. cache-only / no-S3 — `node.cc` (Range) + `dingo_cache.cc` (startup)
```cpp
// node.cc — DEFINE_bool(kv_cache_only, false, "KV: Range miss=NotFound, no S3; allow no-S3 start");
Status CacheNode::Range(... ) {
  auto st = RetrieveCache(ctx, bctx, offset, length, buffer);   // node.cc:236
  if (st.IsNotFound()) {
    if (FLAGS_kv_cache_only) return st;                         // short-circuit, no S3
    st = RetrieveStorage(ctx, bctx, offset, length, buffer, block_length);
  }
  return st;
}
// CacheNode ctor: kv_cache_only ? storage_client_pool_ = make_shared<NullStorageClientPool>()
//                              : StorageClientPoolImpl(mds_client_)
// where NullStorageClientPool::GetStorageClient returns Status::NotSupport.
```
```cpp
// dingo_cache.cc:56-69 — only enforce enable_stage when NOT kv_cache_only:
if (!FLAGS_kv_cache_only && !FLAGS_enable_stage) { ...return -1; }
```

## 5. DingofsTransport (route via dingofs PeerGroup, reuse the portable KVClient)
New `src/cache/kvclient/dingofs_transport.{h,cc}` implementing
`dingofs::cache::kv::Transport` over `remotecache::RemoteBlockCacheImpl`. It
translates the portable `BlockKey`→`pb::cache::BlockContext` and wraps the raw
value bytes in an `IOBuffer` (zero-copy via `AppendUserData`). It **ignores the
`node` arg** because `RemoteBlockCacheImpl` re-derives the peer from the
`BlockKey` via the MDS PeerGroup Ketama; client-side `ConHash` is harness-only,
so pass a single placeholder member to `KVClient`. Put→`SyncCache`,
Get→`Range`+`CopyTo`, Exist→`Exist`. The explicit namespace/object-key codec and
raw-value contract are shared with `dfkv_hicache.py`. The C ABI remains
`libdfkv.so`; a nanobind module is the optional alternative below.

## 6. nanobind module (optional alt to ctypes) — mirror `sdk/python/CMakeLists.txt`
```cmake
find_package(nanobind CONFIG REQUIRED HINTS "${THIRD_PARTY_INSTALL_PATH}/nanobind/cmake")
nanobind_add_module(pydingofs_kv python/py_kv_client.cc)
target_link_libraries(pydingofs_kv PRIVATE cache_kvclient cache_remotecache
                      cache_cachegroup cache_blockcache cache_common cache_iutil)
install(TARGETS pydingofs_kv DESTINATION dingofs)
```

## 7. dingofs CMake wiring — `src/cache/CMakeLists.txt`
`add_subdirectory(kvclient)`; `kvclient/CMakeLists.txt` builds `cache_kvclient`
from the portable `.cc` + `dingofs_transport.cc`, linking
`cache_remotecache cache_cachegroup cache_blockcache cache_common cache_iutil`.

## 8. RDMA (deferred for the dingo-cache fusion path)
Out of scope for the brpc/dingo-cache fusion path (no env). The standalone
`dfkv_server` already has a native-verbs RDMA transport. A future fusion-path
transport must preserve the same explicit namespace, canonical object key, and
opaque raw-value contract.
