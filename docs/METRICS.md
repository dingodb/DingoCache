# dfkv 可观测性（Metrics / 集群视图）

> v1.4.0 起。所有指标均为**新增**、对数据面零性能影响（热路径仅 relaxed 原子；延迟为 1/64 采样；HTTP 在独立端口/线程）。默认不开端口 → 行为与旧版一致。

> **分布式追踪（traces）** 是另一条独立的连接器侧能力（按慢请求 / 采样 / 失败上报 span，经 OTLP `/v1/traces`），见 [tracing.md](tracing.md)。本文只覆盖 metrics。

## 1. 抓取端点

| 进程 | 开关 | 端点 |
|---|---|---|
| `dfkv_server` | `--metrics-port <p>`（缺省=关） | `GET /metrics`、`GET /healthz` |
| `dfkv_mds` | `--metrics-port <p>`（缺省=关） | `GET /metrics`、`GET /healthz` |

身份标签：`dfkv_server --id <node> --group <g>` → 所有序列带 `{node,group}`；不设则**无标签**（向后兼容）。Prometheus 抓取自带 `instance`/`job`。

```bash
dfkv_server --dir /mnt/d1,/mnt/d2 --port 12000 --rdma-port 12001 \
            --id gpu1-0001 --group glm --mds 10.0.0.1:9400 --advertise 10.0.0.11:12000 \
            --metrics-port 9100
curl -s 127.0.0.1:9100/metrics
```

## 2. 集群 / 环视图（CLI）

> **v1.8.0 起**：`dfkvctl ring` 多一列 `INFO` = 各节点在注册/心跳时自报的
> `ver=…,engine=…,disks=…,cap=…,ram=…,rdma=…`（运行时真相，非 flag 意图）。
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
| 指标 | 类型 | 含义 |
|---|---|---|
| `dfkv_build_info{version,transport}` | gauge | 版本 + 构建传输（rdma/tcp），恒为 1 |
| `dfkv_uptime_seconds` | gauge | 启动至今秒数 |
| `dfkv_cache_put_total` / `cache_hit_total` / `cache_miss_total` | counter | PUT / GET 命中 / GET 未命中 |
| `dfkv_exist_hit_total` / `exist_miss_total` | counter | Exist 命中 / 未命中 |
| `dfkv_bytes_written_total` / `bytes_read_total` | counter | 读写字节 |
| `dfkv_accepts_total` | counter | 累计 TCP accept |
| `dfkv_open_connections` | gauge | 当前打开连接数 |
| `dfkv_evictions_total` / `evicted_bytes_total` | counter | 淘汰对象数 / 字节 |
| `dfkv_errors_total{op,status}` | counter | 失败 op（put/get io、invalid） |
| `dfkv_objects` / `used_bytes` / `disks` | gauge | 对象数 / 占用 / 盘数 |
| `dfkv_disk_used_bytes{disk}` / `dfkv_disk_objects{disk}` | gauge | 每盘占用 / 对象 |
| `dfkv_op_latency_seconds{op}` | histogram | **1/64 采样**的 **get / put / exist** 服务端延迟（50µs–100ms 桶）。`op="exist"` 是 exist handler 体延迟（Contains + IsCached 的锁），L3 预取停滞时先查它的尾——慢 exist 卡住预取决策；serve-loop 排队（大 GET 挡在同连接的 exist 前）是另一回事，靠客户端 control-lane QP 隔离规避 |

RDMA 构建额外（折叠进同一 /metrics）：
| `dfkv_rdma_completions_total` / `completion_errors_total` | counter | RDMA 请求完成 / 错误完成 |
| `dfkv_rdma_active_conns` | gauge | 当前服务中的 RDMA 连接 |
| `dfkv_rdma_v1_conns_opened_total` / `v2_conns_opened_total` | counter | server 累计打开的 v1 / v2 连接；滚动升级和回滚验收的路径真值 |
| `dfkv_rdma_v2_put_writes_total` / `v2_get_writes_total` | counter | server 实际收到的 `WRITE_WITH_IMM` PUT / 实际发出的 RDMA WRITE GET payload |
| `dfkv_rdma_recv_segment_bytes` / `dfkv_rdma_recv_segment_free_bytes` | gauge | v2 process-wide receive segment 总字节 / 当前未租 lease 字节；free 持续接近 0 = 新 data QP 将回退 v1 |
| `dfkv_rdma_recv_segment_registered_rails` | gauge | 成功把共享 segment 注册到共享 PD 的 rail 数；应等于显式白名单内可用 rail 数 |
| `dfkv_rdma_v2_ready` | gauge | 至少一个 rail 完成共享 segment 注册；0 = 本进程只能接 v1 |
| `dfkv_uring_reads_total` / `dfkv_uring_init_fallbacks_total` | counter | io_uring 路径真实提交的读数（**>0 = 路径确实激活**，外部可证）/ 想用 uring 但 ring 初始化失败静默回退同步的连接数（>0 = 配了没生效，查内核/权限） |
| `dfkv_rdma_idle_reclaims_total` | counter | 空闲超时回收的连接数 |

SGEngine 兼容入口（对应 flag 未启用时整组不存在）：
| 指标 | 类型 | 含义 |
|---|---|---|
| `dfkv_sgengine_tcp_info{wire="v1",key_domain="sgengine-v1"}` | gauge | 旧 TCP frontend 已启用，恒为 1 |
| `dfkv_sgengine_tcp_accepts_total` / `open_connections` / `requests_total` | counter / gauge | 旧 TCP 连接与请求流量 |
| `dfkv_sgengine_tcp_rejected_ops_total` | counter | 在隔离边界拒绝的 discovery/MDS/control op |
| `dfkv_sgengine_rdma_info{wire="v1",key_domain="sgengine-v1"}` | gauge | 旧 RDMA frontend 已启用且强制 v1，恒为 1 |
| `dfkv_sgengine_rdma_*` / `dfkv_sgengine_uring_*` | 与 native 同名尾缀 | 兼容 RDMA listener 的独立连接、完成、错误和 read-path 指标；不会与 native `dfkv_rdma_*` 混算 |
| `dfkv_sgengine_rdma_rejected_ops_total` | counter | 在旧 RDMA 边界拒绝的 discovery/MDS/control op |

> **v2 上线判据**：新 client + 新 server 车队应看到 client/server 两侧
> `v2_conns_opened_total` 增长，PUT/GET write counter 随负载增长，
> `v2_ready=1` 且 `recv_segment_free_bytes` 有容量余量。若 server 的
> `v1_conns_opened_total` 在全新车队仍增长，先查 free bytes、segment 注册日志、
> client/server 强制协议 env 和旧二进制；不要只凭 `DFKV_RDMA=1` 认定走了 v2。

读侧 convoy 合并（v1.35+，`DFKV_READ_COALESCE=1` 时才有增量；恒零 = 开关没生效）：
| `dfkv_read_coalesce_leaders_total` | counter | 经 coalescer 登记并执行的读（同步 leader + uring flight 完成各计一次;>0 = 合并路径确实在环内） |
| `dfkv_read_coalesced_total` | counter | 从在途同键读直接取到数据的 follower 数（窗口内合并吸收量） |
| `dfkv_read_coalesce_recur_total` | counter | 命中复现指纹（tombstone）的读——漂移超窗后的晋升证据（v2 驻留窗 `DFKV_READ_COALESCE_RECUR_MS`,默认 1000ms） |
| `dfkv_read_coalesce_timeouts_total` | counter | follower 等待超时回退自读的次数（`DFKV_READ_COALESCE_TIMEOUT_MS`,默认 500ms;**健康态应恒 0**,持续非零 = leader 连接异常死亡或盘读时延超阈） |

slab 引擎内部（**仅 `--store-engine slab` 时输出**；file 引擎无此系列）：
| 指标 | 类型 | 含义 |
|------|------|------|
| `dfkv_slab_dio_write_fallback_total` / `dfkv_slab_dio_read_fallback_total` | counter | direct 模式下回退 buffered 的写/读——**非零升高 = page cache 悄悄回来了**（对齐条件被破坏），direct 部署重点盯 |
| `dfkv_slab_table_sync_total` | counter | slots.tbl fdatasync 周期数（`DFKV_SLAB_TABLE_SYNC_MS`，默认 100ms；限定崩溃复活毒化窗口） |
| `dfkv_slab_extent_steals_total` / `dfkv_slab_extent_returns_total` | counter | 跨 class extent 抢占（伴随驱逐，容量失衡信号）/ 全空 extent 主动回池（无损再平衡） |
| `dfkv_slab_deferred_removes_total` | counter | 被在飞 I/O 延迟执行的 Remove |
| `dfkv_slab_inflight_keys` / `dfkv_slab_prep_holds` | gauge | 锁外 I/O 在飞 key 数 / 未释放的异步 prep 持有数（持续增长 = 泄漏） |
| `dfkv_slab_reclaimed_total` | counter | 后台回收线程预驱逐的 slot 数（`DFKV_SLAB_RECLAIM_MS`，默认 50ms）——持续为 0 且 PUT 延迟高 = 回收被关或池未满，先查 `--slab-reclaim-ms` |
| `dfkv_slab_rebalanced_total` | counter | 回收线程从冷 class 搬给热 class 的 extent 数（类再平衡）——换模型/尺寸迁移期应看到增长，稳态应静止 |

PUT 准入门（**仅 `--put-inflight-limit > 0` 时输出**）：
| `dfkv_put_busy_total` | counter | 被准入门以 kCacheFull 快速拒绝的 PUT（受控 miss，替代深队列尾延迟） |

RAM 热层（**仅 `DFKV_RAM_TIER=1` 时输出**；关时无此系列，向后兼容）：
| 指标 | 类型 | 含义 |
|------|------|------|
| `dfkv_ram_hit_total` / `dfkv_ram_miss_total` | counter | GET 命中 RAM / 未命中落盘（命中率 = hit/(hit+miss)） |
| `dfkv_ram_put_total` | counter | 写直通进 RAM 的 PUT 数 |
| `dfkv_ram_put_bypass_total` | counter | **背压**：arena 满（flush 落后）→ PUT 旁路直写盘，非零即 flush 跟不上 |
| `dfkv_ram_promoted_total` | counter | 读晋升（v1.35+,需 `DFKV_READ_COALESCE=1`）：带 convoy 证据（扇入或复现指纹）的整值冷读以 born-durable 身份直入 arena——不进 flushq、零刷盘成本、随时可逐;健康态应跟随 `dfkv_read_coalesce_recur_total` |
| `dfkv_ram_flushed_total` / `dfkv_ram_flush_dropped_total` | counter | RAM slot 落盘转 DURABLE / flush 多次失败后丢弃 |
| `dfkv_ram_evictions_total` | counter | RAM slot 容量压力淘汰数（含内联与后台回收两路） |
| `dfkv_ram_reclaimed_total` | counter | 其中由后台回收线程预驱逐的数量（`DFKV_RAM_RECLAIM_MS`，默认 10ms；flush 积压 >4096 时自动歇拍，此计数暂停属预期） |
| `dfkv_ram_rebalanced_total` | counter | RAM 层类再平衡搬动的 extent 数（增长阶段不受 flush 积压歇拍影响——从冷 donor 搬 durable extent 恰是 flush-gated 时唯一能扩收速的动作） |
| `dfkv_ram_objects` / `dfkv_ram_flush_backlog` | gauge | 当前 RAM 常驻块 / 待 flush（未 DURABLE）队列深度 |

> 关键运维信号：COLD `load_get_avg_ms` 骤降 + `dfkv_ram_hit_total` 上升 = RAM 热层生效；`dfkv_ram_put_bypass_total` 或 `dfkv_ram_flush_backlog` 持续升高 = flush 落盘带宽不足，需扩 flush 或降 PUT 速率（见 [docs/ARCHITECTURE.md](ARCHITECTURE.md) §6 背压）。

### 3.2 MDS（`dfkv_mds` /metrics）

每环汇总（**v1.10.0 起**；scrape 时 MDS 现场 range 一次 etcd，数值来自各节点心跳携带的 STA1 统计，新鲜度≈心跳周期 10s。全部为 **gauge 语义**——节点重启会使 `_sum` 回落，速率分析请用节点级 counter）：
| 指标 | 含义 |
|------|------|
| `dfkv_mds_group_nodes{group}` | 该环成员数（带标签版；旧无标签 `dfkv_mds_members` 保留不动） |
| `dfkv_mds_group_capacity_bytes` / `_used_bytes{group}` | 环总容量 / **环水位**（direct 模式下 df 已失真，此为唯一真值） |
| `dfkv_mds_group_objects{group}` | 环内常驻块数 |
| `dfkv_mds_group_hits_sum` / `_misses_sum{group}` | 环级命中率 = hits/(hits+misses) |
| `dfkv_mds_group_evictions_sum` / `_puts_sum{group}` | 容量压力 / 写入量 |
| `dfkv_mds_group_put_busy_sum{group}` | 准入门拒绝总数（过载信号） |
| `dfkv_mds_group_dio_fallbacks_sum{group}` | direct 模式 buffered 回退总数（**>0 = page cache 悄悄回来了**，舰队级告警位） |
| `dfkv_mds_group_ram_used_bytes` / `_ram_hits_sum{group}` | RAM 热层水位 / 命中 |
| `dfkv_mds_group_stats_missing{group}` | 无 STA1 上报的成员数（滚动升级进度/掉队检测） |
| `dfkv_mds_group_version_skew{group}` | 去重版本数，**>1 = 版本漂移** |

对应 CLI：`dfkvctl stats --mds <eps> --group <g>`（每节点表格+汇总行，数据一跳来自 MDS 不触节点）/ `--all`（kListGroups 枚举全部环）。深钻仍用 `dfkvctl stat --all`（逐节点全量 /metrics）。

原有计数：
| 指标 | 类型 | 含义 |
|---|---|---|
| `dfkv_mds_register_requests_total` / `keepalives_total` | counter | 注册 / 心跳 |
| `dfkv_mds_list_requests_total` | counter | ListMembers 次数 |
| `dfkv_mds_lease_grants_total` | counter | etcd lease 授予 |
| `dfkv_mds_etcd_errors_total` | counter | etcd I/O 失败 |
| `dfkv_mds_members` | gauge | 上次 List 返回的成员数 |

### 3.3 客户端（SGLang 插件 /metrics，经 prometheus_client）
插件后台轮询线程读 C 客户端快照（`client_stats_poll_s`，默认 10s，0=关）并镜像为带 `{tp_rank}` 的 Counter：
| 指标 | 含义 |
|---|---|
| `dfkv_client_ops_served_total` | 收到响应的 op 数 |
| `dfkv_client_io_errors_total` | 客户端观察到的传输失败 |
| `dfkv_client_unhealthy_skips_total` | 因 peer 熔断短路的 op |
| `dfkv_client_peer_marked_bad_total` / `peer_recovered_total` | peer 熔断 / 恢复切换 |

插件直接暴露（per-batch）：
| `dfkv_client_set_calls/pages/ok_pages/bytes_total{tp_rank}` | set 量 |
| `dfkv_client_get_calls/pages/hit_pages/bytes_total{tp_rank}` | get 量 |
| `dfkv_client_set_seconds{tp_rank}` / `get_seconds{tp_rank}` | batch 调用耗时直方图 |

C 客户端快照里还含传输级（RDMA 构建）：
`dfkv_rdma_client_conns_opened_total`、
`dfkv_rdma_client_v1_conns_opened_total` / `dfkv_rdma_client_v2_conns_opened_total`、
`dfkv_rdma_client_v2_put_writes_total` / `dfkv_rdma_client_v2_get_writes_total`、
`dfkv_rdma_client_mr_regions`、
`dfkv_rdma_client_adhoc_user_mr_total`（生产注册池路径应稳定为 0）和
`dfkv_rdma_client_rail_conns_total{dev}`（NUMA/轮转选轨分布）。

### 3.4 连接器车队指标（三连接器 OTLP **push**，opt-in）
§3.3 是 SGLang 插件本地 `/metrics`（**pull**）。此外三个连接器（vLLM `dfkv-vllm` / LMCache `dfkv-connector` / SGLang HiCache `dfkv_hicache.py`）可把聚合后的运行指标经 **OTLP 主动推送**到中心 Collector→Prometheus→Grafana，用于"车队级"按实例/类型观测。
- **opt-in**：`DFKV_METRICS_ENABLED=1` + `OTEL_EXPORTER_OTLP_ENDPOINT`；关时零开销。默认导出器 `DFKV_METRICS_EXPORTER=stdlib`（纯标准库，**零第三方依赖**），可选 `otel`（OpenTelemetry SDK）。完整配置与各引擎接法见 [`../deploy/observability/CONNECTOR-USAGE.md`](../deploy/observability/CONNECTOR-USAGE.md)。
- **公共标签**：`connector_type`（`hicache`/`lmcache`/`vllm`）、`connector_id`（实例标识）、`version`（连接器包版本，滚动升级可见）。

| 指标 | 类型 | 标签 | 含义 |
|---|---|---|---|
| `dfkv_connector_ops_total` | counter | `op` | 各 op 次数（→ 命中率） |
| `dfkv_connector_keys_total` | counter | `op` | 各 op 涉及的 key 数 |
| `dfkv_connector_bytes_total` | counter | `op` | 各 op 字节量（→ 吞吐） |
| `dfkv_connector_op_seconds` | histogram | `op` | 各 op 延迟分布 |
| `dfkv_connector_op_max_seconds` | gauge | `op` | 各 op 周期内峰值延迟 |
| `dfkv_connector_info` / `_info_ratio` | gauge | — | 实例信息 / 命中比 |
| `dfkv_client_peer_latency_avg_seconds` / `_max_seconds` | gauge | `peer` | **逐 dfkv server 节点延迟**（诊断慢节点/慢路径，如跨机房） |
| `dfkv_client_peer_latency_seconds_count` / `_sum` | counter | `peer` | 逐 peer 延迟采样数 / 总和 |

> 逐 peer 延迟由 C++ 客户端主动探测（`DFKV_PROBE_INTERVAL_MS`）：SGLang HiCache 开 metrics 即自动开；vLLM/LMCache 需手动 `export DFKV_PROBE_INTERVAL_MS=5000`。

### 3.5 vLLM 连接器客户端健康（`dfkv-vllm` /metrics，经 prometheus_client，pull）
§3.4 是 OTLP **push**（opt-in）；此外 vLLM 连接器后台轮询线程读同一 C 客户端快照（`DFKV_CLIENT_STATS_POLL_S`，默认 15s，0=关），把**环/MDS 健康**镜像为带 `{tp_rank}` 的 Gauge，直接落在 vLLM 自带 `/metrics` 上（**无需开 telemetry**）。这样"空环（写无处可去、ok=0）/ MDS 不可达"在 scrape 上可见，而非只在客户端日志里——正是一次生产事故（静默空环）的教训。
| 指标 | 类型 | 标签 | 含义 |
|---|---|---|---|
| `vllm:dfkv_client_ring_members` | gauge | `tp_rank` | 客户端看到的环成员数（`0` = 空环） |
| `vllm:dfkv_client_mds_reachable` | gauge | `tp_rank` | 上次发现轮询 MDS 是否可达（`1`/`0`） |
| `vllm:dfkv_client_mds_unreachable_polls_total` | gauge | `tp_rank` | 无法连上 MDS 的发现轮询次数（镜像 C 端计数器） |
| `vllm:dfkv_client_transport_info` | gauge | `tp_rank`,`transport` | 恒为 `1`，`transport` 标 live 传输（`rdma`/`tcp`） |

> 源快照名（C 端无 `vllm:` 前缀）为 `dfkv_client_ring_members` / `dfkv_client_mds_reachable` / `dfkv_client_mds_unreachable_polls_total`（PR#207），与 SGLang HiCache §3.3 同源；vLLM 侧加 `vllm:` 前缀入连接器命名空间。

## 4. 性能保证

- 热路径计数：`std::atomic` relaxed `fetch_add`，零锁零分配（与既有模式一致）。
- 延迟：1/64 采样（`Sampler` 掩码），仅采样到的 op 读 vDSO 时钟。
- HTTP `/metrics`：独立端口 + 线程，仅 scrape 时把原子读成文本。
- 客户端计数搭在 `PeerHealth` 既有锁段上（无新增锁）；per-peer 仅错误路径（罕见）。
- 插件轮询线程睡眠态、按间隔触发，不在 per-batch/per-page 路径。
- 验证：每 PR 跑 RDMA Soft-RoCE loopback + ThreadSanitizer 全绿。
