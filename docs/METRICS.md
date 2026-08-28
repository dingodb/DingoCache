# dfkv 可观测性（Metrics / 集群视图）

> 本页是 **v2.11.1** 的稳定指标契约；表内列出本版本全部公开 metric family，
> 并标明按构建、引擎或开关才出现的条件族。旧版本可能缺少后续增加的指标，
> 不得把“序列不存在”直接解释为 0。`/metrics` 生成仍在独立端口/线程，
> 热路径计数为 relaxed 原子、延迟按 1/64 采样；不开端口时不启动 HTTP。

> **分布式追踪（traces）** 是独立的连接器侧能力（OTLP `/v1/traces`），
> 见 [tracing.md](tracing.md)。本文只覆盖 metrics。

## 1. 抓取端点

| 进程 | 开关 | 端点 |
|---|---|---|
| `dfkv_server` | `--metrics-port <p>`（缺省=关） | `GET /metrics`、`GET /healthz`、`GET /readyz` |
| `dfkv_mds` | `--metrics-port <p>`（缺省=关） | `GET /metrics`、`GET /healthz`、`GET /readyz` |

身份标签：`dfkv_server --id <node> --group <g>` → 节点数据面序列带
`{node,group}`；registrar 健康序列显式带 `{node,group}`。Prometheus 抓取自带
`instance`/`job`。`/healthz` 反映当前 store/RAM terminal health；`/readyz`
还要求本地 startup 完成以及（配置 MDS 时）首次注册成功。首次注册 latch 后，
瞬时 heartbeat 丢失不会清 readiness；用下述 heartbeat 指标单独告警。未配置
MDS 的 node 仅等待本地 startup 和动态 storage health。
IB placement eligibility is independent of these HTTP checks. Only an
RDMA-enabled binary with an active listener emits the server rail families.
At least one healthy initialized rail keeps the node eligible: a partial loss
reports fewer active rails but `/healthz`, `/readyz`, and placement remain
online. Losing the final healthy rail makes `dfkv_server_ring_eligible=0`
immediately; 0→nonzero recovery waits the configured sample streak.
`dfkv_mds` 的语义不同：`/healthz` 是纯进程 liveness，**不 probe etcd**——
否则一次秒级 etcd 抖动会让 kubelet 同时重启全部 MDS 副本，CrashLoop 比抖动
更久并引发 lease 集体过期。`/readyz` 保留 etcd 依赖检查，但经 TTL-debounced
probe（`DFKV_MDS_PROBE_CACHE_MS`，默认 2500 ms 内复用上次结果，0=恢复逐请求
现场 probe）：not-ready 副本每 TTL 最多重探一次，kubelet 抓取节奏不会放大成
etcd 读负载。运行期依赖丢失时 `/readyz` 从 200 变 503，scheduler 必须用它
摘流；etcd 恢复后同一 MDS 进程自动回到 200。

监听加固 knobs（均 env，resolved 后进 config dump；两进程的 metrics HTTP 还
支持 `--metrics-bind <addr>` 绑定监听地址，缺省 0.0.0.0）：

| knob | 默认 | 作用 |
|---|---|---|
| `DFKV_METRICS_FIRST_REQ_MS` | 30000（0=关，硬上限 3600000） | metrics HTTP 首行请求 absolute deadline（accept 起算），防 drip/silent 连接占 handler |
| `DFKV_METRICS_MAX_CONNS` | 64（硬上限 4096） | metrics HTTP 并发连接上限，超限直接关闭 |
| `DFKV_MDS_FIRST_REQ_MS` | 30000（0=关，硬上限 3600000） | `dfkv_mds` 控制面首帧 absolute deadline |
| `DFKV_MDS_MAX_CONNS` | 4096 | `dfkv_mds` 并发连接上限 |
| `DFKV_MDS_PROBE_CACHE_MS` | 2500（0=逐请求现场 probe，钳制上限 600000） | `dfkv_mds` `/readyz` etcd probe TTL 去抖 |
| `DFKV_TCP_FIRST_REQ_MS` | 30000（0=关，硬上限 3600000） | `dfkv_server` 数据面首帧 absolute deadline；resolved 值见 `dfkv_tcp_first_req_ms` |

```bash
dfkv_server --dir /mnt/d1,/mnt/d2 --port 12000 --rdma-port 12001 \
            --id gpu1-0001 --group glm --mds 10.0.0.1:9400 --advertise 10.0.0.11:12000 \
            --metrics-port 9100
curl -s 127.0.0.1:9100/metrics
```

## 2. 集群 / 环视图（CLI）

> **v1.8.0 起**：`dfkvctl ring` 多一列 `INFO` = 各节点在注册/心跳时自报的
> `ver=…,engine=…,wr=…,disks=…,cap=…,ram=…,rdma=…,qd=…`（运行时 resolved
> 真相，非 flag 意图；`wr=` 是 slab 的 direct/buffered resolved 写模式，`qd=`
> 是 server 端协商后的 pipeline depth）。
> 全环版本/引擎审计一条命令完成（抓"静默跳升级/引擎不一致"）。`-` = 节点还是
> 旧版未上报（本身就是版本信号）。信息不参与环 epoch（变更不会触发客户端重建环）。
> 生效条件：server 与 MDS 都 ≥ v1.8.0（老 MDS 会丢弃该扩展字段）。

```bash
dfkvctl ring     --mds <ep,...> --group <g>   # 成员表 + 一致性哈希环 vnode 分布/占比
dfkvctl stat --all --mds <ep,...> --group <g> # 逐节点指标 + 集群聚合（容量/对象/命中率）
dfkvctl stat <ip:port>                        # 单节点原始 /metrics 文本（旧用法不变）
```

## 3. 指标清单

### 3.1 cache 节点（`dfkv_server` /metrics）
下表及其条件子表构成 v2.11.1 的完整公开 family 清单。带 `{disk}`、`{dev}`、
`{tenant_hash}`、`{slot_size}` 的序列，其 label 值来自启动时有界集合，不能由请求输入扩张。

| 指标 | 类型 | 含义 |
|---|---|---|
| `dfkv_build_info{version,transport,engine,write_mode[,node,group]}` | gauge | 版本 + 构建传输（rdma/tcp）+ resolved 存储引擎；slab 的 `write_mode` 为 `direct`/`buffered`，显式 file 为 `n/a`；设置 `--id`/`--group` 时追加 `node`/`group` label；恒为 1 |
| `dfkv_uptime_seconds` | gauge | 启动至今秒数 |
| `dfkv_storage_healthy` | gauge | 当前 disk group 与显式请求 RAM tier 的 terminal health；0 时 `/healthz`、`/readyz` 均为 503 |
| `dfkv_server_startup_complete` / `dfkv_server_mds_registration_ready` / `dfkv_server_healthy` / `dfkv_server_ready` | gauge | startup、本次首次 MDS 注册门、本地动态 health、三者合取；`dfkvctl stat --all` 用这组稳定 gauge 判定节点健康 |
| `dfkv_cache_put_total` / `dfkv_cache_hit_total` / `dfkv_cache_miss_total` | counter | PUT / GET 命中 / GET 未命中 |
| `dfkv_exist_hit_total` / `dfkv_exist_miss_total` | counter | Exist 命中 / 未命中 |
| `dfkv_remove_ok_total` / `dfkv_remove_miss_total` | counter | Remove 删掉了块 / 目标本就不存在（partial save 清理等路径的诊断分型） |
| `dfkv_bytes_written_total` / `dfkv_bytes_read_total` | counter | 读写字节 |
| `dfkv_accepts_total` | counter | 累计 TCP accept |
| `dfkv_open_connections` | gauge | 当前打开连接数 |
| `dfkv_tcp_max_connections` / `dfkv_tcp_io_timeout_seconds` | gauge | cache TCP handler 硬准入上限 / socket I/O 超时的 resolved 配置 |
| `dfkv_tcp_first_req_ms` | gauge | 首帧 absolute deadline 的 resolved 配置（`DFKV_TCP_FIRST_REQ_MS`，默认 30000，0=关；防 drip 连接占 handler 槽位） |
| `dfkv_tcp_rejected_connections_total` | counter | 达 handler 上限后拒绝的新 TCP 连接；增长表示 silent/flood 或连接池规模超过预算 |
| `dfkv_mds_registration_latched{group,node}` | gauge | server 首次 MDS 注册是否完成；成功后保持 1 |
| `dfkv_mds_first_registration_timeout_ms{group,node}` | gauge | 首次注册 deadline 的 resolved 值；cache daemon 默认 60000，合法配置范围 1000–600000 |
| `dfkv_mds_first_registration_timed_out{group,node}` | gauge | deadline 是否已到期；到期后进程立即进入 fail-closed shutdown 并退出 1 |
| `dfkv_mds_heartbeat_healthy{group,node}` | gauge | 首次注册后最近一次 heartbeat 是否成功 |
| `dfkv_mds_heartbeat_failures_consecutive{group,node}` / `dfkv_mds_heartbeat_failures_total{group,node}` | gauge / counter | 连续 / 累计 heartbeat 失败 |
| `dfkv_mds_last_success_age_seconds{group,node}` | gauge | 最近成功注册或 heartbeat 的年龄 |
| `dfkv_evictions_total` / `dfkv_evicted_bytes_total` | counter | 淘汰对象数 / 字节 |
| `dfkv_errors_total{op,status}` | counter | 失败 op（put/get io、invalid） |
| `dfkv_objects` / `dfkv_used_bytes` / `dfkv_disks` | gauge | 对象数 / 占用 / 盘数 |
| `dfkv_disk_used_bytes{disk}` / `dfkv_disk_objects{disk}` | gauge | 每盘占用 / 对象 |
| `dfkv_tenant_default_quota_bytes` | gauge | 本节点未显式列出 tenant 的 quota；0=无限 |
| `dfkv_tenant_quota_rejections_total` | counter | 本节点所有 tenant 的 `kQuotaExceeded` item 总数 |
| `dfkv_tenant_quota_limit_bytes{tenant_hash}` | gauge | 配置文件中该 16 位小写 hex tenant hash 的 per-node limit；0=无限 |
| `dfkv_tenant_used_bytes{tenant_hash}` | gauge | 该显式配置 tenant 在本节点的 committed payload bytes |
| `dfkv_tenant_quota_rejections_by_hash_total{tenant_hash}` | counter | 该显式配置 tenant 在本节点的 quota rejection |
| `dfkv_op_latency_seconds{op}` | histogram | **1/64 采样**的 **get / put / exist** 服务端延迟（50µs–60s 桶，另有 `+Inf`）。`op="exist"` 是 exist handler 体延迟（Contains + IsCached 的锁），L3 预取停滞时先查它的尾；serve-loop 排队（大 GET 挡在同连接的 exist 前）是另一回事，靠客户端 control-lane QP 隔离规避 |

Tenant 标签**只**来自启动时有界配置文件中的 16-hex hash；默认 quota 命中的
未列 tenant 不生成动态 label。容量余量为
`dfkv_tenant_quota_limit_bytes - dfkv_tenant_used_bytes`（limit=0 时不告警）。
`rate(dfkv_tenant_quota_rejections_total[5m]) > 0` 表示 quota 拒绝；不要与
`dfkv_put_busy_total`（写入门 `kCacheFull`）混为一类。

建议告警：cache daemon 的首次注册由
`--mds-registration-timeout-ms` / `DFKV_MDS_REGISTRATION_TIMEOUT_MS`
（默认 60000 ms，1000–600000，严格解析）有界；到期退出 1 而不是无限 unready。
`dfkv_mds_heartbeat_healthy == 0` 持续 2 个 heartbeat 或
`dfkv_mds_last_success_age_seconds > 30` 用于**成功注册后的** MDS 降级；
此时 cache `/readyz` 保持 200，恢复后续租。`rate(dfkv_tcp_rejected_connections_total[5m]) > 0`
触发时同步看 `dfkv_open_connections / dfkv_tcp_max_connections`。

IB rail 告警先看 active cardinality，再看 eligibility：
`dfkv_server_rdma_rails_active < dfkv_rdma_rails_initialized` 表示 `PARTIAL`
带宽；只要 active ≥1，节点仍在 placement。`dfkv_server_ring_eligible == 0`
表示最后健康 rail 已丢失、该节点全部 cache capacity 已摘除。逐轨定位用
`dfkv_server_ib_device_healthy{device}`。这些查询在 TCP-only server 或未启动
RDMA listener 的 RDMA build 上不会返回序列，这是预期行为；不得用
`or vector(0)` 把缺失补成 unhealthy。告警“应有但缺失”必须关联明确的
RDMA-listener scrape inventory。

默认 `DFKV_RDMA_HEALTH_RECOVERY_SAMPLES=3`。从非零健康轨降到零立即清 eligibility；
从零恢复到非零才要求连续成功的本地健康采样机会。通过后，全部轨健康为
`ACTIVE`，部分健康为 `PARTIAL`。heartbeat 名义间隔 10 s 时三次机会通常约
20–30 s（取决于相位），还要叠加 registrar/MDS 发布延迟；这不是上界，也无需
重启。部分掉轨 9→8 不走该恢复门，因为节点从未失去最后健康轨。

四种 rail 口径不能混用：

| 口径 | 权威来源 | 解释 |
|---|---|---|
| configured input / monitored topology | startup log `configured=N`; `dfkv_server_rdma_rails_configured` | log 值是显式列表去重数（auto 为 0）；gauge 是 health monitor 的固定 resolved rail 数（auto 含选出的 rail） |
| initialized/resolved | startup log `initialized=N`; `dfkv_rdma_rails_initialized`; `dfkv_rdma_recv_segment_registered_rails` | 完成 anchor、共享 segment 和全部启动 MR 的固定 topology，auto 含选出的 rail |
| startup-ACTIVE | startup log `ACTIVE=N` / `inactive=...` | 仅启动瞬间快照 |
| current healthy | `dfkv_server_rdma_rails_active`; `sum(dfkv_server_ib_device_healthy)` | 当前 ACTIVE+LinkUp 数；至少 1 即可 placement |

仅在 **RDMA-enabled binary 实际启动 RDMA listener** 时额外输出（折叠进同一
`/metrics`）；TCP-only binary 或未启动 RDMA listener 时下列 family 不存在：

| 指标 | 类型 | 含义 |
|---|---|---|
| `dfkv_server_ring_eligible` | gauge | 当前是否有至少一条通过门控的 initialized healthy rail；最后一条丢失立即为 0，0→非 0 需 recovery streak |
| `dfkv_server_rdma_rails_configured` / `dfkv_rdma_rails_initialized` / `dfkv_server_rdma_rails_active` | gauge | health monitor 固定 resolved 数 / 成功 anchor 初始化数 / 当前健康数；auto 模式均包含选出的 rail，无 `device` label |
| `dfkv_server_ib_device_healthy{device}` | gauge | 每个固定 initialized device 当前是否 `query_ok && ACTIVE && LinkUp`；部分为 0 不再等价于整节点退环 |
| `dfkv_rdma_completions_total` / `dfkv_rdma_completion_errors_total` | counter | RDMA 请求完成 / 错误完成 |
| `dfkv_rdma_active_conns` | gauge | 当前服务中的 RDMA 连接 |
| `dfkv_rdma_v2_conns_opened_total` | counter | server 累计打开的 v2 连接 |
| `dfkv_rdma_v2_put_writes_total` / `dfkv_rdma_v2_get_writes_total` | counter | server 实际收到的 `WRITE_WITH_IMM` PUT / 实际发出的 RDMA WRITE GET payload |
| `dfkv_rdma_recv_segment_bytes` / `max_bytes` / `chunks` | gauge | receive-pool 当前提交量 / hard budget / 已提交 chunk 数 |
| `dfkv_rdma_recv_segment_used_bytes` / `free_bytes` | gauge | 已提交 chunk 中 lease 占用 / 空闲字节 |
| `dfkv_rdma_recv_segment_largest_free_range_bytes` | gauge | 任一 chunk 最大连续 free range |
| `dfkv_rdma_recv_segment_growths_total` / `growth_failures_total` | counter | 启动后 chunk 增长成功 / 因预算或分配失败 |
| `dfkv_rdma_recv_segment_allocation_failures_total` | counter | grow 后仍无法满足的最终 allocation |
| `dfkv_rdma_pull_connections` / `dfkv_rdma_legacy_connections` | gauge | 当前 pull-read / legacy responder-write connection 数 |
| `dfkv_rdma_connection_bytes{class=\"data|control\"}` | gauge | data/control connection 当前 lease 字节；应随 adaptive class 而非 logical max 增长 |
| `dfkv_rdma_recv_segment_registered_rails` | gauge | 成功注册初始 receive chunk 的 rail 数；后续 chunk 按使用 rail 惰性注册 |
| `dfkv_rdma_v2_ready` | gauge | 初始 receive chunk 与 rail anchor 是否就绪 |
| `dfkv_rdma_segment_evictions_total` / `dfkv_rdma_idle_reclaims_total` | counter | hard budget 压力淘汰 / idle timeout 回收连接 |
| `dfkv_uring_reads_total` / `dfkv_uring_init_fallbacks_total` | counter | io_uring 路径实际读 / 初始化失败回退同步连接数 |
| `dfkv_rdma_rail_active_conns{dev}` | gauge | 每个本地 HCA 上当前连接数 |
| `dfkv_rdma_rail_completions_total{dev}` / `dfkv_rdma_rail_completion_errors_total{dev}` | counter | 每 rail 请求完成 / 错误完成 |
| `dfkv_rdma_rail_put_writes_total{dev}` / `dfkv_rdma_rail_put_bytes_total{dev}` | counter | 每 rail 收到的 PUT one-sided writes / payload bytes |
| `dfkv_rdma_rail_get_writes_total{dev}` / `dfkv_rdma_rail_get_bytes_total{dev}` | counter | 每 rail 发出的 GET one-sided writes / payload bytes |

> **v2 上线判据**：显式 topology 必须
> `configured == initialized == dfkv_rdma_recv_segment_registered_rails`。
> 启动 `ACTIVE` 只是快照；运行期以
> `dfkv_server_rdma_rails_active` 和逐设备 healthy 为准。active≥1 时
> `dfkv_server_ring_eligible=1`。同时要求 `dfkv_rdma_v2_ready=1`、
> `committed <= max`、growth/allocation failures 不增长。

读侧 convoy 合并与直读晋升（`DFKV_READ_COALESCE=1` 时才有增量；恒零 = 开关没生效）：
| `dfkv_read_coalesce_leaders_total` | counter | 经 coalescer 登记并完成的读（同步 leader + io_uring flight 各计一次；>0 = 合并路径确实在环内） |
| `dfkv_read_coalesced_total` | counter | 被在途同键读吸收的 follower 数；直读晋升 follower 等发布后从 RAM 获取独立 send pin，不复制 leader payload |
| `dfkv_read_coalesce_recur_total` | counter | staged fallback flight 命中复现指纹的诊断计数（`DFKV_READ_COALESCE_RECUR_MS`，默认 1000ms）；direct-to-RAM flight 不创建指纹 |
| `dfkv_read_coalesce_timeouts_total` | counter | follower 等待超时回退自读的次数（`DFKV_READ_COALESCE_TIMEOUT_MS`，默认 500ms；**健康态应恒 0**，持续非零 = leader 连接异常死亡或盘读时延超阈） |

slab 引擎内部（**resolved engine=slab 时输出**——按运行时实际引擎判定，不设 flag/env 的默认也命中；file 引擎无此系列）：
| 指标 | 类型 | 含义 |
|------|------|------|
| `dfkv_slab_dio_write_fallback_total` / `dfkv_slab_dio_read_fallback_total` | counter | direct 模式下回退 buffered 的写/读——**非零升高 = page cache 悄悄回来了**（对齐条件被破坏），direct 部署重点盯 |
| `dfkv_slab_table_sync_total` | counter | slots.tbl fdatasync 周期数（`DFKV_SLAB_TABLE_SYNC_MS`，默认 100ms；限定崩溃复活毒化窗口） |
| `dfkv_slab_extent_steals_total` / `dfkv_slab_extent_returns_total` | counter | 跨 class extent 抢占（伴随驱逐，容量失衡信号）/ 全空 extent 主动回池（无损再平衡） |
| `dfkv_slab_cold_steals_total` | counter | 命中"全局最冷 extent"护栏的跨 class 抢占（donor 全部内容都冷于 `DFKV_SLAB_COLD_STEAL_WINDOW` 窗口才走此路，满环自噬护栏；0=窗口关闭） |
| `dfkv_slab_watermark_evictions_total` | counter | high crossing 后持续到 low 的主动整 extent 驱逐数 |
| `dfkv_slab_watermark_extent_clears_total` | counter | watermark 使用单次连续 slots.tbl clear 的 extent 数；应与 watermark eviction 同步 |
| `dfkv_slab_watermark_ticks_total` / `dfkv_slab_watermark_active` | counter / gauge | bounded eviction tick 数 / hysteresis 是否仍在 drain |
| `dfkv_slab_watermark_last_tick_us` / `max_tick_us` | gauge | 最近 / 进程最大 watermark 锁区间微秒数 |
| `dfkv_slab_watermark_max_extents_per_tick` | gauge | 每盘每 tick 配置的 extent 上限 |
| `dfkv_slab_table_rebuilt_objects` / `dfkv_slab_rebuild_scanned_bytes` / `dfkv_slab_rebuild_scan_chunks` / `dfkv_slab_rebuild_sparse_ranges` / `dfkv_slab_rebuild_mmap_scans` | gauge | 上次启动冷升从 slots.tbl 恢复的对象数 / 扫描字节 / chunk / sparse range / mmap 的表数（冷升目录规模诊断） |
| `dfkv_slab_rebuild_corrupt_records_total` / `dfkv_slab_rebuild_rejected_records_total` / `dfkv_slab_rebuild_sequential_fallbacks_total` | counter | 启动 rebuild 丢弃的损坏记录 / 因几何不安全被清除的合法记录 / sparse seek 退化为顺序扫描的次数（任一非零都值得看日志） |
| `dfkv_slab_deferred_removes_total` | counter | 被在飞 I/O 延迟执行的 Remove |
| `dfkv_slab_inflight_keys` / `dfkv_slab_prep_holds` | gauge | 锁外 I/O 在飞 key 数 / 未释放的异步 prep 持有数（持续增长 = 泄漏） |
| `dfkv_slab_reclaimed_total` | counter | 后台回收线程预驱逐的 slot 数（`DFKV_SLAB_RECLAIM_MS`，默认 50ms）——持续为 0 且 PUT 延迟高 = 回收被关或池未满，先查 `--slab-reclaim-ms` |
| `dfkv_slab_rebalanced_total` | counter | 回收线程从冷 class 搬给热 class 的 extent 数（类再平衡）——换模型/尺寸迁移期应看到增长，稳态应静止 |
| `dfkv_slab_batched_writes_total` / `dfkv_slab_uring_write_batches_total` | counter | 进入 batched store visit 的 payload 写数 / io_uring 单次 submit 轮数 |
| `dfkv_slab_bind_wipes_total` | counter | extent 绑定新 class 前完成的旧 slot-grid 清理 |
| `dfkv_slab_metadata_io_errors_total` / `dfkv_slab_unclean_resets_total` | counter | slots.tbl/epoch 持久化失败 / 冷升时丢弃 dirty epoch |
| `dfkv_slab_eviction_record_clears_total` / `dfkv_slab_record_writes_total` | counter | allocator 复用前持久清除记录 / 已提交 slots.tbl record 写数 |
| `dfkv_slab_capacity_bytes` / `dfkv_slab_allocated_bytes` / `dfkv_slab_payload_bytes` / `dfkv_slab_internal_fragmentation_bytes` | gauge | 物理 payload 容量 / 已占 slot / 逻辑 payload / 内部碎片 |
| `dfkv_slab_allocator_objects` / `dfkv_slab_committed_objects` | gauge | allocator resident（含未提交写）/ reader-visible 已提交对象 |
| `dfkv_slab_classes` / `dfkv_slab_bound_extents` / `dfkv_slab_pool_extents` | gauge | live class 数 / 已绑定 extent / 未绑定共享池 extent |
| `dfkv_slab_class_extents{slot_size}` / `dfkv_slab_class_resident_objects{slot_size}` / `dfkv_slab_class_free_slots{slot_size}` | gauge | 每个 live size class 的 extent、resident、可立即分配 slot |
| `dfkv_slab_class_allocated_bytes{slot_size}` / `dfkv_slab_class_useful_bytes{slot_size}` / `dfkv_slab_class_fragmentation_bytes{slot_size}` | gauge | 每 class 的占用、有效 payload、内部碎片 |
| `dfkv_slab_class_read_heat{slot_size}` / `dfkv_slab_class_put_total{slot_size}` | gauge / counter | 每 class 的衰减读热度 / 累计 PUT |
| `dfkv_slab_failed_disks` / `dfkv_slab_healthy` | gauge | fail-closed disk 数 / 全部 slab disk 是否健康 |

PUT 准入门（**仅 `--put-inflight-limit > 0` 时输出**）：
| `dfkv_put_busy_total` | counter | 被准入门以 kCacheFull 快速拒绝的 PUT（受控 miss，替代深队列尾延迟） |

RAM 热层（**仅 `DFKV_RAM_TIER=1` 时输出**；关时无此系列，向后兼容）：
| 指标 | 类型 | 含义 |
|------|------|------|
| `dfkv_ram_hit_total` / `dfkv_ram_miss_total` | counter | GET 命中 RAM / 未命中落盘（命中率 = hit/(hit+miss)） |
| `dfkv_ram_put_total` | counter | 写直通进 RAM 的 PUT 数 |
| `dfkv_ram_put_bypass_total` | counter | **背压**：arena 满（flush 落后）→ PUT 旁路直写盘，非零即 flush 跟不上 |
| `dfkv_ram_promoted_total` | counter | 整值冷读晋升：优先让 O_DIRECT 直接读入隐藏 arena reservation，成功后以 born-durable 身份发布；staged fallback 则读后复制晋升。不进 flushq、零重复刷盘、随时可驱逐；健康冷→热运行中首次读增长，随后应转为 `dfkv_ram_hit_total` 增长且磁盘字节不再增加 |
| `dfkv_ram_flushed_total` / `dfkv_ram_flush_dropped_total` | counter | RAM slot 落盘转 DURABLE / flush 多次失败后丢弃 |
| `dfkv_ram_healthy` | gauge | RAM flusher 未发生 terminal failure 时为 1；0 会动态摘除 readiness |
| `dfkv_ram_flush_threads` | gauge | shard 最小值调整后的实际 flusher 数（不是原始请求值） |
| `dfkv_ram_evictions_total` | counter | RAM slot 容量压力淘汰数（含内联与后台回收两路） |
| `dfkv_ram_reclaimed_total` | counter | 其中由后台回收线程预驱逐的数量（`DFKV_RAM_RECLAIM_MS`，默认 10ms；flush 积压 >4096 时自动歇拍，此计数暂停属预期） |
| `dfkv_ram_rebalanced_total` | counter | RAM 层类再平衡搬动的 extent 数（增长阶段不受 flush 积压歇拍影响——从冷 donor 搬 durable extent 恰是 flush-gated 时唯一能扩收速的动作） |
| `dfkv_ram_objects` / `dfkv_ram_flush_backlog` | gauge | 当前 RAM 常驻块 / 待 flush（含已出队、正在 flush，均尚未 DURABLE）的块数 |
| `dfkv_ram_budget_bytes` / `dfkv_ram_arena_bytes` / `dfkv_ram_large_budget_bytes` | gauge | RAM 层总预算（arena+dedicated）/ arena 固定预留 / 总预算内大对象 dedicated 预留 |
| `dfkv_ram_used_bytes` / `dfkv_ram_large_used_bytes` | gauge | 已用（常驻 slot + dedicated 分配）/ 其中 dedicated 大对象已用——与上组预算行做水位比 |

> 关键运维信号：COLD `load_get_avg_ms` 骤降 + `dfkv_ram_hit_total` 上升 = RAM 热层生效；`dfkv_ram_put_bypass_total` 或 `dfkv_ram_flush_backlog` 持续升高 = flush 落盘带宽不足，需扩 flush 或降 PUT 速率（见 [docs/ARCHITECTURE.md](ARCHITECTURE.md) §6 背压）。

### 3.2 MDS（`dfkv_mds` /metrics）
`dfkv_mds` 自身的 scheduler readiness 不是启动 latch：`/readyz` 经
TTL-debounced etcd probe 判定（`DFKV_MDS_PROBE_CACHE_MS`，默认 2500 ms 内复用
上次结果，0=逐请求现场 probe），probe 频率由 TTL 去抖决定而非 kubelet 抓取
节奏；`/healthz` 不探 etcd，恒反映进程存活。因此支持 200 → 503 → 200 原地
恢复。


每环汇总（**v1.10.0 起**；scrape 时 MDS 现场 range 一次 etcd，数值来自各节点心跳携带的 STA1 统计，新鲜度≈心跳周期 10s。全部为 **gauge 语义**——节点重启会使 `_sum` 回落，速率分析请用节点级 counter）：
| 指标 | 含义 |
|------|------|
| `dfkv_mds_group_nodes{group}` | 该环成员数（带标签版；旧无标签 `dfkv_mds_members` 保留不动） |
| `dfkv_mds_group_ring_eligible_nodes{group}` / `dfkv_mds_group_degraded_nodes{group}` | 当前可参与 placement 的节点数 / 因 IB health 被排除但 lease 仍在线的节点数 |
| `dfkv_mds_group_capacity_bytes{group}` / `dfkv_mds_group_used_bytes{group}` | 环总容量 / **环水位**（direct 模式下 df 已失真，此为唯一真值） |
| `dfkv_mds_group_objects{group}` | 环内常驻块数 |
| `dfkv_mds_group_hits_sum{group}` / `dfkv_mds_group_misses_sum{group}` | 环级命中率 = hits/(hits+misses) |
| `dfkv_mds_group_evictions_sum{group}` / `dfkv_mds_group_puts_sum{group}` | 容量压力 / 写入量 |
| `dfkv_mds_group_put_busy_sum{group}` | 准入门拒绝总数（过载信号） |
| `dfkv_mds_group_dio_fallbacks_sum{group}` | direct 模式 buffered 回退总数（**>0 = page cache 悄悄回来了**，舰队级告警位） |
| `dfkv_mds_group_ram_used_bytes{group}` / `dfkv_mds_group_ram_hits_sum{group}` | RAM 热层水位 / 命中 |
| `dfkv_mds_group_stats_missing{group}` | 无 STA1 上报的成员数（滚动升级进度/掉队检测） |
| `dfkv_mds_group_version_skew{group}` | 去重版本数，**>1 = 版本漂移** |
| `dfkv_mds_group_clients{group}` | 当前注册的 inference connector/consumer 实例数；client lease 到期后自动下降 |

对应 CLI：`dfkvctl topology --mds <eps> --group <g>` 通过 `kListTopology` 展示全部在线节点和 HLT1 逐设备健康。HLT1 是 Members 的已有可选扩展（placement eligibility + repeated device name/port_state/phys_state/query_ok），不是独立 rail protocol。`MembersTopologyEpoch` 对 member id/device name 稳定排序并包含地址/placement binding 与全部 health 字段；因此 rail health 变化不必重建 Ketama 环，仍会更新 transport。`dfkvctl stats` 只聚合 eligible placement；`PARTIAL` 节点仍在其中，只有零 active 的 `DEGRADED` 节点进入 degraded 计数。

原有计数：
| 指标 | 类型 | 含义 |
|---|---|---|
| `dfkv_mds_ready` | gauge | TTL-debounced etcd probe；与 `/readyz` 使用同一 cache，1=可服务 membership |
| `dfkv_build_info{version,role="mds"}` | gauge | MDS 二进制版本；恒为 1 |
| `dfkv_mds_register_requests_total` / `dfkv_mds_keepalives_total` | counter | 注册 / 心跳 |
| `dfkv_mds_list_requests_total` | counter | ListMembers 次数 |
| `dfkv_mds_lease_grants_total` | counter | etcd lease 授予 |
| `dfkv_mds_etcd_errors_total` | counter | etcd I/O 失败 |
| `dfkv_mds_members` | gauge | 上次 List 返回的成员数 |
| `dfkv_mds_client_register_requests_total` / `dfkv_mds_client_keepalives_total` / `dfkv_mds_client_list_requests_total` | counter | consumer 注册 / 心跳 / ListClients |
| `dfkv_mds_clients` | gauge | 上次 ListClients 返回的 consumer 数 |
| `dfkv_mds_etcd_requests_total{op}` / `dfkv_mds_etcd_request_errors_total{op}` | counter | etcd HTTP 请求 / 传输或 HTTP status 失败，`op=lease_grant|lease_keepalive|put|range|metrics_range`；`metrics_range` 是每次 MDS scrape 同时用于 readiness 与 group 聚合的唯一 range，和业务控制面流量分开计数 |
| `dfkv_mds_etcd_request_duration_seconds{op}` | histogram | 同五类 etcd HTTP 请求延迟（50µs–60s + `+Inf`）；用于区分本地控制面排队、常规 etcd 慢响应和 scrape 自身的聚合 range |
| `dfkv_mds_local_member_leases` / `dfkv_mds_local_client_leases` | gauge | 本 MDS 进程当前缓存的 member/client lease shortcut 数；只是 etcd 权威状态的有界优化 |
| `dfkv_mds_local_member_leases_pruned_total` / `dfkv_mds_local_client_leases_pruned_total` | counter | 因多个 TTL 未在本 MDS 使用而丢弃的本地 shortcut；churn 下增长正常，下次心跳会 fresh grant+Put |
| `dfkv_mds_legacy_frames_total` | counter | 以 v1.x 旧控制面帧（42/10 字节帧）服务的请求数；仅在 `DFKV_MDS_ACCEPT_LEGACY=1` 的混合代际迁移期出现，**归零后应撤掉该 env** |

### 3.3 客户端（SGLang 插件 `/metrics` 与原生 C 快照）
SGLang 后台线程每 `client_stats_poll_s`（默认 10s，0=关）读取一次 C 客户端快照，
把下列有界 family 镜像到插件 `/metrics` 并追加 `{tp_rank}`；request path 不做
Prometheus 调用。vLLM 使用同一 allowlist，区别只是在 family 前加 `vllm:`，见 §3.5。

| 指标 | 类型 | 含义 |
|---|---|---|
| `dfkv_client_stats_snapshot_success` / `dfkv_client_stats_snapshot_timestamp_seconds` / `dfkv_client_stats_snapshot_errors_total` | gauge / gauge / counter | 最新 poll 是否成功 / 最后成功 Unix 时间 / 失败累计；失败时保留 last-good 数据并显式置 success=0 |
| `dfkv_client_transport_info{transport}` | gauge | live transport identity（`rdma`/`tcp`），恒为 1 |
| `dfkv_client_ops_served_total` / `dfkv_client_io_errors_total` / `dfkv_client_health_checks_total` | counter | 收到响应 / 传输失败 / endpoint health check |
| `dfkv_client_unhealthy_skips_total` / `dfkv_client_busy_suppressed_total` | counter | peer 熔断短路 / busy 后主动抑制请求 |
| `dfkv_client_peer_marked_bad_total` / `dfkv_client_peer_recovered_total` | counter | peer 熔断 / 恢复切换 |
| `dfkv_client_peer_errors_total{peer}` | counter | 逐 endpoint 错误；peer 集合在原生层硬限 4096 |
| `dfkv_client_ring_members` / `dfkv_client_mds_reachable` / `dfkv_client_mds_unreachable_polls_total` | gauge / gauge / counter | 当前环节点数 / 最近发现轮询是否可达 MDS / 不可达轮询累计 |
| `dfkv_client_dedup_hits_total` / `dfkv_client_dedup_fetches_total` / `dfkv_client_dedup_wait_hits_total` / `dfkv_client_dedup_wait_timeouts_total` | counter | host rendezvous 命中、真实 fetch、等待后命中、等待超时 |
| `dfkv_client_gpu_dedup_hits_total` / `dfkv_client_gpu_dedup_fetches_total` / `dfkv_client_gpu_dedup_wait_hits_total` / `dfkv_client_gpu_dedup_wait_timeouts_total` | counter | 同上 CUDA IPC rendezvous |

原生 `KVClient` 的收敛操作指标（`op="put|get|exist|remove"`）：

| 指标 | 含义 |
|---|---|
| `dfkv_client_op_requests_total{op}` | 公开 scalar/batch/SG 调用次数；每次调用严格加 1 |
| `dfkv_client_op_keys_total{op}` | 调用提交的 key 数；TCP fan-out、RDMA pipeline、重试和 rendezvous 不重复计数 |
| `dfkv_client_op_hits_total{op}` | put/remove 确认成功数、get 命中数、exist 存在数；shm/CUDA rendezvous 命中也计入 |
| `dfkv_client_op_bytes_total{op}` | 成功移动的 payload 字节 |
| `dfkv_client_op_latency_seconds{op}`（C 快照 histogram）/ `dfkv_client_op_max_seconds{op}`（镜像 gauge） | 完整公开调用延迟（含 rendezvous 等待和有界重试）/ lifetime max；prometheus_client 不重复注册外部累计 histogram，插件另用下述 per-batch histogram |

路由为空、peer cooldown、I/O 失败等早退仍计入 request/key（hit 为 0）。
因此这些指标可直接校验请求守恒，不应与 transport 请求数或 dedup fetch 数相等。

插件直接暴露的调用层 family（全部带 `{tp_rank}`）：

| 指标 | 类型 | 含义 |
|---|---|---|
| `dfkv_client_set_calls_total` / `dfkv_client_set_pages_total` / `dfkv_client_set_ok_pages_total` / `dfkv_client_set_bytes_total` | counter | v1 set 调用、提交页、成功页、字节 |
| `dfkv_client_get_calls_total` / `dfkv_client_get_pages_total` / `dfkv_client_get_hit_pages_total` / `dfkv_client_get_bytes_total` | counter | v1 get 调用、请求页、命中页、字节 |
| `dfkv_client_set_seconds` / `dfkv_client_get_seconds` | histogram | v1 batch set/get 延迟 |
| `dfkv_client_set_v2_calls_total` / `dfkv_client_set_v2_pages_total` / `dfkv_client_set_v2_ok_pages_total` / `dfkv_client_set_v2_bytes_total` | counter | side-pool v2 set 调用、提交页、成功页、字节 |
| `dfkv_client_get_v2_calls_total` / `dfkv_client_get_v2_pages_total` / `dfkv_client_get_v2_hit_pages_total` / `dfkv_client_get_v2_bytes_total` | counter | side-pool v2 get 调用、请求页、命中页、字节 |
| `dfkv_client_set_v2_seconds` / `dfkv_client_get_v2_seconds` | histogram | v2 batch set/get I/O 延迟 |
| `dfkv_client_exist_v2_calls_total` / `dfkv_client_exist_v2_probe_pages_total` / `dfkv_client_exist_v2_hit_pages_total` | counter | v2 prefix probe 调用、候选页、可用 hit-prefix 页 |

C 客户端快照还含传输级指标（RDMA 构建）：

| 指标 | 类型 | 含义 |
|---|---|---|
| `dfkv_rdma_client_conns_opened_total` | counter | 累计打开的 RDMA client QP |
| `dfkv_rdma_client_v2_put_writes_total` / `dfkv_rdma_client_v2_get_writes_total` | counter | 实际发出的 v2 one-sided PUT / GET |
| `dfkv_rdma_client_mr_regions` / `dfkv_rdma_client_mr_registered_bytes` | gauge | 所有 active rail 已成功 anchor 后才发布的 host pool MR 区域数 / 声明字节数；任一 rail 失败时两者保持上次成功值 |
| `dfkv_rdma_client_mr_registration_rejections_total` | counter | 非法 range、rail anchor 或 `ibv_reg_mr` 失败而未发布（已回滚）的声明次数 |
| `dfkv_rdma_client_adhoc_user_mr_total` / `dfkv_rdma_client_transient_user_mr_active` | counter / gauge | pool 外实际注册累计 / 当前仍存活的一次性 MR；公开调用返回后 active 必须回到调用前基线 |
| `dfkv_rdma_client_pool_mr_registrations_total` / `dfkv_rdma_client_pool_mr_registration_failures_total` | counter | shared-PD 上真实 `ibv_reg_mr` 次数 / 单 rail 显式 pool 注册失败；包含最终回滚的尝试 |
| `dfkv_rdma_client_pool_mr_active_registrations` | gauge | 进程内仍有 endpoint 引用的 shared-PD MR generations；扩容成功后 anchor/空闲 endpoint 立即释放旧代，在飞旧 endpoint 到下次 acquire/close 才释放，确保旧 range 不中断 |
| `dfkv_rdma_client_max_block_seen_bytes` / `dfkv_rdma_client_declared_max_block_bytes` / `dfkv_rdma_client_connection_min_block_bytes` | gauge | 实际请求高水位 / 逻辑安全上限 / adaptive QP 最小 class |
| `dfkv_rdma_client_oversize_rejects_total` | counter | 分配、注册或发帖前因超过声明上限而拒绝的操作 |
| `dfkv_rdma_client_v2_probe_attempts_total` / `dfkv_rdma_client_v2_probe_failures_total` | counter | 必选 v2 bootstrap probe 尝试 / 失败 |
| `dfkv_rdma_client_stale_pool_retries_total` | counter | pooled QP 失败后改用 fresh connection 的重试 |
| `dfkv_rdma_client_cross_rail_retries_total` | counter | 无 label；client-local `kRailFailure` 后真正启动的 cross-rail logical retry。GET 可重试未完成 work；PUT 只有 request 未 post 才计入 |
| `dfkv_rdma_client_cross_rail_retry_successes_total` | counter | 无 label；上述 retry 最终成功的 logical operation 数 |
| `dfkv_rdma_client_cross_rail_retry_exhausted_total` | counter | 无 label；上述 retry 的第二次物理尝试仍失败 |
| `dfkv_rdma_client_completion_timeouts_total` | counter | 消耗完一次绝对 completion-window deadline 的窗口；部分完成不重置预算 |
| `dfkv_rdma_client_endpoint_budget{kind}` / `qp_budget{kind}` / `wr_slot_budget{kind}` / `registered_slot_bytes_budget{kind}` | gauge | `kind=used|limit` 的进程级资源占用/上限；registered bytes 按每 endpoint 的 server receive + pull 两个 arena 计量 |
| `dfkv_rdma_client_resource_budget_raises_total` | counter | 预算自适应放大次数（随 ring 采纳，只增不减；任一预算 env 显式设置时恒为 0） |
| `dfkv_rdma_client_admission_failures_total` | counter | 本进程预算饥饿导致的操作失败（返回 kResourceExhausted，对端从未被拨号、不进冷却）；应恒为 0，>0 说明预算仍小于扇出需求 |
| `dfkv_rdma_client_depth_refunds_total` | counter | 服务端 clamp depth 后按协商值退还 WR/注册字节预算的连接数（首连按上限探测，后续按学习值建连） |
| `dfkv_rdma_client_topology_nodes` | gauge | 最近一次喂给预算自适应的 ring 规模（0=从未提示或自适应关闭） |
| `dfkv_rdma_client_peer_topology_updates_total` | counter | 无 label；接纳的新 peer immutable topology generation 数；同 generation 重复通知不增加 |
| `dfkv_rdma_client_no_compatible_rail_total` | counter | 无 label；configured tiers 因 absent/incomplete peer topology 或最高可用交集为空而返回 `kNoCompatibleRail` 的次数；不计 peer health failure |
| `dfkv_rdma_client_stale_generation_reaps_total` | counter | 无 label；因 Conn 的 peer generation 落后而拒绝 take/repool/keepalive、并经 single-owner lifecycle 回收的 endpoint 数 |
| `dfkv_rdma_client_keepalive_attempts_total` / `dfkv_rdma_client_keepalive_successes_total` / `dfkv_rdma_client_keepalive_failures_total` | counter | idle pooled QP 的保活尝试 / 成功 / 失败并退役连接；仅 `DFKV_RDMA_KEEPALIVE_MS>0` 时增长 |
| `dfkv_rdma_client_rail_conns_total{dev}` / `dfkv_rdma_client_rail_selections_total{dev}` | counter | 每 rail 新连接 / 准入分布 |
| `dfkv_rdma_client_rail_inflight{dev}` / `dfkv_rdma_client_rail_credits_available{dev}` | gauge | 当前已租 / 可用 request credits |
| `dfkv_rdma_client_rail_credits_exhausted_total{dev}` | counter | 因 local candidate credit 不足跳过次数 |
| `dfkv_rdma_client_rail_errors_total{dev}` / `dfkv_rdma_client_rail_consecutive_errors{dev}` | counter / gauge | **仅本地** device open、QP transition、MR register/refresh、post/CQ verbs API 失败和 local WC 累计 / 当前连续值 |
| `dfkv_rdma_client_endpoint_errors_total{dev}` | counter | peer bootstrap/不可达/probe/protocol/receive-segment/decode、remote/RNR/retry/timeout/flush WC；credit 已归还，**不惩罚或恢复 rail**。没有 local-error WC 的 completion deadline 也属 endpoint，不触发 cross-rail retry |
| `dfkv_rdma_client_rail_quarantines_total{dev}` / `dfkv_rdma_client_rail_quarantined{dev}` / `dfkv_rdma_client_rail_recovery_probe{dev}` / `dfkv_rdma_client_rail_recoveries_total{dev}` | counter / gauge / gauge / counter | 连续 3 次 client-local failure 后隔离（默认 cooldown 5000 ms）/ 隔离状态 / cooldown 到期后唯一真实 probe / probe 成功恢复；probe local failure 重启 cooldown，endpoint failure 只释放 probe ownership |
| `dfkv_rdma_client_numa_fallbacks_total{reason="caller_unknown\|no_local_rail"}` | counter | `DFKV_RDMA_NUMA=1` 无法建立 local mask 而回退全部 enabled rails |
| `dfkv_rdma_cq_completions_total` / `dfkv_rdma_cq_errors_total` | counter | client CQ 完成 / 错误完成 |
| `dfkv_rdma_client_pipeline_depth` | gauge | 握手解析后的有效 pipeline depth |
| `dfkv_rdma_client_pool_connections{lane,state=\"idle|active\",dev}` | gauge | endpoint 按最后/当前 operation lane、ownership state 和 rail 分解；active 不含已隔离连接 |
| `dfkv_rdma_client_peer_connections{peer,dev}` | gauge | 当前 live endpoint 按 MDS stable peer identity 与本地 rail 分解；连接销毁后 series 删除，cardinality 受 endpoint budget 上限约束 |
| `dfkv_rdma_client_pool_limit` | gauge | 每 peer/pool 的 idle retention 上限（默认 8） |

TCP 构建或 TCP fallback 的 C 快照 family（同样由插件镜像）：

| 指标 | 类型 | 含义 |
|---|---|---|
| `dfkv_transport_pool_connections` / `dfkv_transport_pool_inflight` | gauge | 当前 pooled 连接 / 在飞 acquire |
| `dfkv_transport_pool_selections_total` | counter | routed endpoint 的 pooled connection 选择数 |
| `dfkv_transport_pool_retirements_total{reason}` | counter | 连接退休数；`reason=idle|error` |
| `dfkv_transport_pool_backoff_endpoints` | gauge | 当前阻止连接增长的 endpoint 数 |
| `dfkv_transport_pool_backoff_events_total` / `dfkv_transport_pool_backoff_suppressed_total` | counter | 进入 backoff / backoff 中被 fast-fail 的 acquire |

RDMA client transport family 只在 RDMA runtime 使用相应传输并由 client snapshot/
connector poller 导出时可见；TCP-only 进程缺失是预期行为。所有新 peer-topology
family 均是无 label process counter，避免 peer 地址造成无界时序；`{dev}` 仍只
来自进程启动时的固定 local rail 集合。`kNoCompatibleRail` 是本地兼容性结果，
不增加 `dfkv_client_ops_served_total`、`dfkv_client_io_errors_total`、
`dfkv_client_peer_marked_bad_total` 或 cooldown。endpoint 和 local-rail health
仍按各自既有 family 分开告警。

**retry 语义与告警**：

- `cross_rail` counter 只统计 client-local `kRailFailure` 后真正启动的跨轨 retry；
  pooled endpoint staleness仍由 `stale_pool_retries_total` 单独计数。每个 logical
  operation 捕获一个 peer generation，物理尝试最多 2 次。
- GET/batch/SG 保留已完成 item/window 的 byte-exact staged 结果，fresh 第二次只
  提交未完成集合。PUT 仅在明确 request 尚未 post 时允许换轨；post 后 completion
  不明的 PUT 不 cross-rail replay，避免扩大 ambiguous write 语义。
- Conn 绑定 peer generation；pool 中旧代 endpoint 永不复用或回池，并通过既有
  single-owner lifecycle 销毁。topology update 不在别的线程直接双重 teardown。
- `Status::kNoCompatibleRail` 是 client-local（不上 wire）、peer-health neutral：
  不增加 served、IO error、marked-bad 或 cooldown。configured tiers 遇 absent/
  incomplete HLT1 或空健康交集时返回它；未配置 tiers 的 legacy homogeneous 模式
  不需要完整 peer topology。
- `rate(dfkv_rdma_client_cross_rail_retry_exhausted_total[5m]) > 0` 或持续
  `rail_quarantined{dev} == 1` 需要处置。workload drain 后 inflight 或 transient
  MR 不归零按资源泄漏告警。

该能力不新增 rail wire protocol：MDS 复用 Members HLT1，transport 数据面仍是
RDMA v2。新 client + 当前 MDS + legacy member 在未配置 tiers 时保持
homogeneous 兼容；configured tiers 对该 member fail closed。legacy MDS 不识别
`kListTopology`，modern poll 失败并保留 last-good ring，rollout 必须 MDS-first，
不能把这类 poll failure 解释为 homogeneous fallback。

### 3.4 连接器车队指标（三连接器 OTLP **push**，opt-in）
§3.3 是进程本地 Prometheus **pull**。vLLM、LMCache、SGLang HiCache 还可把
同一份聚合状态经 OTLP push 到中心 Collector，用一个 dashboard 按实例、引擎、
模型、部署、版本和节点过滤。

- 开关：`DFKV_METRICS_ENABLED=1`；关闭时 recorder 是冻结 no-op singleton，
  不 import OTel，不启动线程。
- 默认 `DFKV_METRICS_EXPORTER=stdlib` 使用纯标准库 **OTLP/HTTP JSON**，
  endpoint 必须是 HTTP receiver（通常 `http://collector:4318`）；自动补
  `/v1/metrics`。把标准 gRPC 端口 `4317` 交给 stdlib exporter 会启动告警，
  Collector 也会因协议不匹配拒绝请求。
- 显式 `DFKV_METRICS_EXPORTER=otel` 才使用 OpenTelemetry SDK，可按 SDK 配置
  gRPC `4317` 或 HTTP `4318`。详细接法见
  [`../deploy/observability/CONNECTOR-USAGE.md`](../deploy/observability/CONNECTOR-USAGE.md)。
- OTLP resource attribute 是 `dfkv.connector_type`、`dfkv.connector_id`、
  `dfkv.host`、`dfkv.pid`、`dfkv.tp_rank`，以及有值时的 `dfkv.version`、
  `dfkv.native_version`、`dfkv.model`、`dfkv.deployment`、
  `dfkv.cache_role`、`dfkv.team`。示例 Collector 开启
  `resource_to_telemetry_conversion` 后，Prometheus label 对应为
  `dfkv_connector_type`、`dfkv_connector_id` 等。
- 同进程多个 connector 共享 recorder/tracer 并按实例引用计数；最后一个 close
  停止 native poller 后同步 final export。失败的窗口最大值和 native counter
  delta 保留到下一次成功推送，不会因 snapshot/reset 顺序丢失。

| 指标 | 类型 | 标签 | 含义 |
|---|---|---|---|
| `dfkv_connector_ops_total` / `dfkv_connector_keys_total` / `dfkv_connector_bytes_total` | counter | `op` | Python connector 完成的调用、key、payload bytes |
| `dfkv_connector_op_seconds` | histogram | `op` | connector 调用延迟 |
| `dfkv_connector_op_max_seconds` | gauge | `op` | 上一导出窗口观察到的最大 connector 调用延迟 |
| `dfkv_connector_info` | gauge | — | connector 活跃心跳，值为 1；身份在 resource labels |
| `dfkv_connector_client_stats_poll_success` / `dfkv_connector_client_stats_poll_errors_total` / `dfkv_connector_client_stats_last_success_unixtime` | gauge / counter / gauge | — | 最新原生快照 poll 是否成功 / 失败累计 / 最后成功时间；失败保留 last-good |
| `dfkv_connector_op_requests_total` / `dfkv_connector_op_keys_total` / `dfkv_connector_op_hits_total` / `dfkv_connector_op_bytes_total` | counter | `op` | C++ KVClient 收敛 chokepoint 的公开调用计数 |
| `dfkv_connector_op_latency_seconds` / `dfkv_connector_op_lifetime_max_seconds` | histogram / gauge | `op` | C++ 完整调用延迟 / lifetime max |
| `dfkv_client_peer_latency_avg_seconds` / `dfkv_client_peer_latency_max_seconds` | gauge | `peer` | 当前 poll 窗口平均值 / lifetime max |
| `dfkv_client_peer_latency_seconds_count` / `dfkv_client_peer_latency_seconds_sum` | counter | `peer` | C++ 累计 peer probe 样本数 / 时长 |
| `dfkv_connector_dedup_hits_total` / `dfkv_connector_dedup_fetches_total` / `dfkv_connector_dedup_wait_hits_total` / `dfkv_connector_dedup_wait_timeouts_total` | counter | — | host rendezvous dedup 漏斗 |
| `dfkv_connector_gpu_dedup_hits_total` / `dfkv_connector_gpu_dedup_fetches_total` / `dfkv_connector_gpu_dedup_wait_hits_total` / `dfkv_connector_gpu_dedup_wait_timeouts_total` | counter | — | CUDA IPC rendezvous 漏斗 |

其余原生状态/传输 family 由 §3.3 的 allowlist **一一**导出，名称映射是公开契约：

| C 快照前缀 | OTLP/Prometheus 前缀 | 范围 |
|---|---|---|
| `dfkv_client_` | `dfkv_connector_` | health、ring、MDS；不含由上表单独转换的 op/dedup/peer |
| `dfkv_rdma_client_` | `dfkv_connector_rdma_` | QP、MR、rail、NUMA、timeout、pipeline |
| `dfkv_rdma_cq_` | `dfkv_connector_rdma_cq_` | CQ completion/error |
| `dfkv_transport_pool_` | `dfkv_connector_transport_pool_` | TCP pool、retirement、backoff |

例如 §3.3 的 `dfkv_rdma_client_completion_timeouts_total` 对应
`dfkv_connector_rdma_completion_timeouts_total`。逐 endpoint
`dfkv_client_peer_errors_total{peer}` 不进入 OTLP：动态 peer 集合虽在 C++ 内有
4096 上限，仍不适合作为全车队中心序列；中心诊断使用固定 rail family 和聚合
peer health counter。

逐 peer 延迟由 C++ 主动探测（`DFKV_PROBE_INTERVAL_MS`）：SGLang HiCache 开
metrics 时自动启用；vLLM/LMCache 需显式配置非零 interval。

### 3.5 vLLM 本地客户端指标（`dfkv-vllm` `/metrics`，pull）
vLLM 的 `DFKV_CLIENT_STATS_POLL_S`（默认 15s，0=关）镜像 §3.3 中**全部**
allow-listed counter/gauge，而不是只镜像 ring/MDS。名称固定为
`vllm:<C快照原名>`，并追加 `{tp_rank}` 与源 family 的 `{op|peer|dev|reason|transport}`。
例如：

| 指标 | 类型 | 含义 |
|---|---|---|
| `vllm:dfkv_client_ring_members` / `vllm:dfkv_client_mds_reachable` | gauge | 空环 / MDS 可达性 |
| `vllm:dfkv_rdma_client_rail_errors_total{tp_rank,dev}` | counter | 每 rank、每 rail 本地错误 |
| `vllm:dfkv_transport_pool_connections{tp_rank}` | gauge | TCP pool 当前连接 |
| `vllm:dfkv_client_stats_snapshot_success` / `vllm:dfkv_client_stats_snapshot_timestamp_seconds` / `vllm:dfkv_client_stats_snapshot_errors_total` | gauge / gauge / counter | poll last-good 健康；无需开启 OTLP telemetry |

### 3.6 成员一致性巡检 textfile 指标

`deploy/dfkv_membership_audit.py --prom-output <path>` 原子写 node_exporter textfile
格式。它是巡检结果，不是 daemon 热路径指标：

| 指标 | 类型 | 含义 |
|---|---|---|
| `dfkv_membership_audit_ok{group}` | gauge | 每个 MDS placement view、etcd 注册、自报 INFO、ring epoch 全部一致时为 1 |
| `dfkv_membership_audit_issues{group,severity}` | gauge | 最近一次巡检的 critical / warning 数 |
| `dfkv_membership_ring_epoch{group}` | gauge | 按 placement 内容计算的 FNV-1a ring epoch（主要用于变更关联；Prometheus float 可能舍入 64-bit 低位） |
| `dfkv_membership_registration_revision{group,node}` | gauge | etcd 中该节点注册 key 的最新 `mod_revision`，可关联停止更新/大面积重注册事件 |

建议 cron/systemd timer 每分钟运行一次，命令本身用 `--timeout 5`；同时对
`dfkv_membership_audit_ok == 0` 或脚本退出非零告警。textfile 有上一次成功文件不代表
本次 timer 成功，必须另外监控 timer exit/文件 mtime；不要只看陈旧的 `ok=1`。
split-brain、MDS/etcd 漂移和自报不一致均由脚本以 `critical` + exit `2` 给出，
JSON `issues[].code/message` 包含具体 endpoint/node 和处置方向。

## 4. 性能保证

- 热路径计数：`std::atomic` relaxed `fetch_add`，零锁零分配（与既有模式一致）。
- 延迟：1/64 采样（`Sampler` 掩码），仅采样到的 op 读 vDSO 时钟。
- HTTP `/metrics`：独立端口 + 线程，仅 scrape 时把原子读成文本。
- 客户端计数搭在 `PeerHealth` 既有锁段上（无新增锁）；per-peer 仅错误路径（罕见）。
- 插件轮询线程睡眠态、按间隔触发，不在 per-batch/per-page 路径。
- 验证：每 PR 跑 RDMA Soft-RoCE loopback + ThreadSanitizer 全绿。
