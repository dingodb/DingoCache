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
Placement eligibility is what the node reports to the MDS on every heartbeat,
and `dfkv_server_ring_eligible` is that report. It is narrower than `/healthz`:
a disk store that failed for good (it then stores and reads nothing more
until the process is restarted) hands its keys to the node's healthy disks,
and the node stays in placement with `dfkv_server_degraded=1` and
`store=degraded` in INFO. Once fewer than `DFKV_DEGRADED_MIN_DISKS` (default
1) healthy disks are left, the node leaves placement on its next heartbeat,
for good, while a RAM tier that dropped an entry it could not flush turns
`/healthz` to 503 but leaves the node, which still serves every request, in
placement. The node stays registered: `dfkvctl ring` shows it as `DEGRADED`
with `store=failed` in INFO until it is restarted or replaced.
IB rail health is the other input, and is independent of the HTTP checks. Only
an RDMA-enabled binary with an active listener emits the server rail families.
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
| `DFKV_MDS_MAX_CONNS` | 4096 | `dfkv_mds` 并发连接上限。N 个连接最多需要 2N+128 个 fd；启动时软上限不足会先在硬上限内提高，仍不足则下调本上限并打 WARN，生效值见 `dfkv_mds_max_conns` |
| `DFKV_RDMA_MAX_PENDING_HANDSHAKES` | 4096（0=不限） | `dfkv_server` RDMA bootstrap 监听上尚未收齐 bootstrap 帧的连接数上限，超限直接关闭；不限制已建立的 RDMA 连接数 |
| `DFKV_MDS_PROBE_CACHE_MS` | 2500（0=逐请求现场 probe，钳制上限 600000） | `dfkv_mds` `/readyz` etcd probe TTL 去抖 |
| `DFKV_MDS_UNWRITTEN_HOLD_MS` | 120000（4×lease TTL；0=关，钳制上限 600000） | `dfkv_mds` 把“心跳已到但没写进 etcd”的 member 视为存活的时长；期间 etcd 视图缺少该 member 则 member list 返回 `kIOError`（见 §3.2）。须大于 MDS 端点数 × 10s 心跳间隔 |
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
> 是 server 端协商后的 pipeline depth）。有盘 store 终态故障、仍在其余盘上服务的节点
> 在末尾追加 `store=degraded`；健康盘少于 `DFKV_DEGRADED_MIN_DISKS` 的节点追加
> `store=failed`，同时以 `DEGRADED` 退出 placement（见 §3.1 `dfkv_server_ring_eligible`）。
> 全环版本/引擎审计一条命令完成（抓"静默跳升级/引擎不一致"）。`-` = 节点还是
> 旧版未上报（本身就是版本信号）。信息不参与环 epoch（变更不会触发客户端重建环）。
> 生效条件：server 与 MDS 都 ≥ v1.8.0（老 MDS 会丢弃该扩展字段）。

```bash
dfkvctl ring     --mds <ep,...> --group <g>   # 成员表 + 一致性哈希环 vnode 分布/占比
dfkvctl ring --json --mds <ep,...> --group <g> # 同一视图的 JSON 文档（给脚本；表格格式不变）
dfkvctl stat --all --mds <ep,...> --group <g> # 逐节点指标 + 集群聚合（容量/对象/命中率）
dfkvctl stat <ip:port>                        # 单节点原始 /metrics 文本（旧用法不变）
```

`ring --json`（`topology --json` 相同）输出一份 JSON 文档，内容与表格一致：

```json
{"group":"g","ring_points":320,"placement_epoch":"17505949231424193949","members":[
{"id":"n1","ip":"10.0.0.1","port":28001,"status":"ACTIVE","eligible":true,"weight":1,"vnodes":160,"share_percent":50.0,"ib_health":"unknown","info":"ver=…,engine=…"},
{"id":"n2","ip":"10.0.0.2","port":28001,"status":"ACTIVE","eligible":true,"weight":1,"vnodes":160,"share_percent":50.0,"ib_health":"unknown","info":"ver=…,engine=…"}
]}
```

- `placement_epoch` 是对 eligible 成员计算的 `MembersEpoch`，也就是 MDS 和客户端用来判断环
  是否变化的那个值。它是 64 位无符号整数，以十进制字符串给出（按 double 读数字的工具放不下
  64 位）。
- `eligible=false` 的成员已注册但不在放置环里（`vnodes` 为 0）；`status`、`ib_health` 与表格
  的 `STATUS`、`IB-HEALTH` 列相同；节点没有自报时 `info` 为空字符串（表格里是 `-`）。
- 字符串字段按字节输出：可打印 ASCII 以外的字节一律写成 `\u00XX`（值即该字节），所以节点
  上报的内容即使不是合法 UTF-8，文档仍是合法 JSON。要还原成文本，把解析出的字符串里每个
  码点当作一个字节，再按 UTF-8 解码（Python：`s.encode("latin-1").decode("utf-8")`）；
  纯 ASCII 的内容不用处理。
- 字段只增不改，使用方应忽略不认识的字段。退出码与表格模式相同：空环或没有 eligible 成员时
  为 1（仍会输出文档），加 `--allow-empty` 为 0。`--json` 只支持 `ring`/`topology`，用在
  其他子命令上以 2 退出。
- `deploy/` 下的运维脚本读这份文档，不解析表格。

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
| `dfkv_server_ring_eligible` | gauge | 本节点在心跳里自报的 placement eligibility：健康盘不少于 `DFKV_DEGRADED_MIN_DISKS`（默认 1），且（有 RDMA listener 时）至少一条健康 rail。健康盘不足后恒为 0 直到进程重启；TCP-only 节点也输出 |
| `dfkv_server_degraded` | gauge | 1：有盘 store 终态故障，节点在其余盘上照常服务、留在 placement（坏盘的 key 由健康盘承接，先未命中、重写后命中）。告警 `DfkvServerDegraded` |
| `dfkv_cache_put_total` / `dfkv_cache_hit_total` / `dfkv_cache_miss_total` | counter | PUT / GET 命中 / GET 未命中 |
| `dfkv_exist_hit_total` / `dfkv_exist_miss_total` | counter | Exist 命中 / 未命中 |
| `dfkv_remove_memory_entries` / `dfkv_remove_memory_covered_seconds` | gauge | 近期删除记忆（`DFKV_REMOVE_MEMORY_MS`，DEPLOY §4c-4，只在打开时输出）：记着的 key 数 / 往回多久内的删除节点都知道（启动以来，或丢掉的最新一条之后，至多保留时长） |
| `dfkv_remove_memory_dropped_total` | counter | 记忆满了丢掉的记录（之后节点知道的时间段变短） |
| `dfkv_removed_within_ages_requests_total` | counter | 回答过的 `kRemovedWithinAges`（op 26，DEPLOY §4c-6）请求数；客户端没开 `DFKV_RING_READBACK_REWRITE` 时保持 0 |
| `dfkv_write_memory_entries` | gauge | 最近 PUT 记忆（DEPLOY §4c-6，`DFKV_WRITE_MEMORY_ENTRIES`）里的 key 数 |
| `dfkv_write_memory_dropped_total` | counter | 最近 PUT 记忆满了丢掉的记录（只是少回读） |
| `dfkv_write_memory_cleared_total{reason}` | counter | 窗口内 `CLOCK_REALTIME`（`realtime`）或 `CLOCK_BOOTTIME`（`boottime`）相对 `CLOCK_MONOTONIC_RAW` 漂移超过 1 秒而清空最近 PUT 记忆的次数（可能停过机；chrony 大幅调整也会触发） |
| `dfkv_removed_within_queries_total{result}` | counter | `kRemovedWithin` 问到的 key，按回答：`removed`、`not_removed`、`unknown` |
| `dfkv_remove_ok_total` / `dfkv_remove_miss_total` | counter | Remove 删掉了块 / 目标本就不存在（partial save 清理等路径的诊断分型） |
| `dfkv_bytes_written_total` / `dfkv_bytes_read_total` | counter | 读写字节 |
| `dfkv_accepts_total` | counter | 累计 TCP accept |
| `dfkv_open_connections` | gauge | 当前打开连接数 |
| `dfkv_tcp_max_connections` / `dfkv_tcp_io_timeout_seconds` | gauge | cache TCP handler 硬准入上限 / socket I/O 超时的 resolved 配置 |
| `dfkv_tcp_first_req_ms` | gauge | 首帧 absolute deadline 的 resolved 配置（`DFKV_TCP_FIRST_REQ_MS`，默认 30000，0=关；防 drip 连接占 handler 槽位） |
| `dfkv_tcp_rejected_connections_total` | counter | 达 handler 上限后拒绝的新 TCP 连接；增长表示 silent/flood 或连接池规模超过预算 |
| `dfkv_tcp_accept_failures_total` | counter | cache TCP 监听 `accept()` 因 fd 或内存耗尽（EMFILE/ENFILE/ENOBUFS/ENOMEM 等非单连接错误）失败的次数；每次失败后退避 100 ms 再试。增长 = 进程 fd 上限或主机内存不足 |
| `dfkv_tcp_thread_start_failures_total` | counter | 因 handler 线程创建失败而拒绝的 cache TCP 连接（进程继续运行）。增长 = 触到线程数上限（systemd `TasksMax`、cgroup `pids.max`）或内存不足 |
| `dfkv_mds_registration_latched{group,node}` | gauge | server 首次 MDS 注册是否完成；成功后保持 1 |
| `dfkv_mds_first_registration_timeout_ms{group,node}` | gauge | 首次注册 deadline 的 resolved 值；cache daemon 默认 60000，合法配置范围 1000–600000 |
| `dfkv_mds_first_registration_timed_out{group,node}` | gauge | deadline 是否已到期；到期后进程立即进入 fail-closed shutdown 并退出 1 |
| `dfkv_mds_heartbeat_healthy{group,node}` | gauge | 首次注册后最近一次 heartbeat 是否成功 |
| `dfkv_mds_fenced{group,node}` | gauge | MDS 以“已 fence”拒绝本节点（`dfkvctl member fence`）；为 1 时节点不在环里，直到 `unfence` |
| `dfkv_mds_heartbeat_failures_consecutive{group,node}` / `dfkv_mds_heartbeat_failures_total{group,node}` | gauge / counter | 连续 / 累计 heartbeat 失败 |
| `dfkv_mds_last_success_age_seconds{group,node}` | gauge | 最近成功注册或 heartbeat 的年龄 |
| `dfkv_evictions_total` / `dfkv_evicted_bytes_total` | counter | 淘汰对象数 / 字节 |
| `dfkv_errors_total{op,status}` | counter | 失败 op（put/get io、invalid） |
| `dfkv_objects` / `dfkv_used_bytes` / `dfkv_disks` | gauge | 对象数 / 占用 / 盘数 |
| `dfkv_disks_failed` / `dfkv_disk_failed{disk}` | gauge | store 已终态故障的盘数 / 每盘是否已终态故障（1 = 故障，直到进程重启） |
| `dfkv_disk_takeover_excluded{disk}` | gauge | 1：本次启动删不掉这块盘上不属于它的 key（WARN），本轮运行里它不再承接坏盘的 key（自己的 key 照常服务）；故障记录保留，下次启动重试 |
| `dfkv_disks_open_dropped_keys{reason}` | gauge | 本次启动因上一轮有盘故障而删掉的 key 数：`failed_disk` = 清空故障盘，`not_home` = 其余盘上不属于本盘的 key（上一轮代坏盘承接的），`unusable_record` = 故障记录读不全或写着本节点没有的目录时清空所有盘。没有故障记录的启动为 0 |
| `dfkv_disk_used_bytes{disk}` / `dfkv_disk_objects{disk}` | gauge | 每盘占用 / 对象 |
| `dfkv_disk_engine_info{disk,engine}` | gauge | 恒为 1：每块盘实际运行的存储引擎（`slab`、`file`、`log`）。`DFKV_STORE_ENGINE=auto`（实验性）按目录选引擎时，各盘可能不同 |
| `dfkv_lse_segments{disk,disk_index}` / `dfkv_lse_segments_free` / `dfkv_lse_segments_sealed` / `dfkv_lse_sealer_queue` / `dfkv_lse_shadows` | gauge | 仅 `log` 引擎盘（实验性）：段总数 / 空闲段（含预开段）/ 已封段 / 待封段 / 未进索引但仍可恢复的副本数 |
| `dfkv_lse_segments_sealed_total` / `dfkv_lse_segments_reclaimed_total` / `dfkv_lse_inline_reclaims_total` | counter | 仅 `log` 引擎盘：封段 / 回收段 / PUT 因无空闲段自行回收的段 |
| `dfkv_lse_inline_reclaim_microseconds_total` / `dfkv_lse_sealer_queue_waits_total` / `dfkv_lse_sealer_queue_wait_microseconds_total` / `dfkv_lse_cleaner_waits_total` / `dfkv_lse_cleaner_wait_microseconds_total` | counter | 仅 `log` 引擎盘：写入路径上的等待。PUT 自行回收段的耗时；写入者（PUT 或搬迁）因待封队列已满而等待封段器的次数 / 耗时；PUT 找不到空间、等待刚启动的清理器的次数 / 耗时（启动后 1 秒内）。用两者之比看平均等待 |
| `dfkv_lse_segment_open_waits_total` / `dfkv_lse_segment_open_wait_microseconds_total` / `dfkv_lse_preopened_taken_total` | counter | 仅 `log` 引擎盘：PUT 等待另一个 PUT 写新段头（段头在锁外写，设计 §6.7）的次数 / 耗时；直接用上预开段（不必写段头）的开段次数 |
| `dfkv_lse_preopen_waits_total` / `dfkv_lse_preopen_wait_microseconds_total` | counter | 仅 `log` 引擎盘：写满当前段的 PUT 等预开段写完段头（一次写）的次数 / 耗时；不等就只能内联回收（只淘汰不搬迁）别的段 |
| `dfkv_lse_records_relocated_total` / `dfkv_lse_relocated_bytes_total` / `dfkv_lse_records_evicted_total` / `dfkv_lse_protected_evictions_total` / `dfkv_lse_relocate_budget_exhausted_total` / `dfkv_lse_victims_skipped_pins_total` | counter | 仅 `log` 引擎盘：清理器搬迁记录 / 字节，淘汰记录，作为最后手段淘汰的受保护记录，因搬迁预算用尽而淘汰的被读过的记录，因 pin 未在时限内释放而跳过的段 |
| `dfkv_lse_cache_full_total` / `dfkv_lse_read_crc_failures_total` | counter | 仅 `log` 引擎盘：回答 `kCacheFull` 的 PUT / 载荷校验失败的读与搬迁读 |
| `dfkv_lse_journal_compactions_total` / `dfkv_lse_journal_integrity_resets_total` / `dfkv_lse_journal_bytes` | counter / counter / gauge | 仅 `log` 引擎盘：元数据日志压缩次数 / 因日志损坏清空存储的次数 / 日志大小 |
| `dfkv_lse_recovery_segments_walked` / `dfkv_lse_recovery_records_rejected` | gauge | 仅 `log` 引擎盘：上次打开时逐块扫描的段 / 因头或载荷校验失败丢弃的记录 |
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
表示该节点全部 cache capacity 已摘除：`dfkv_server_rdma_rails_active == 0`
是最后健康 rail 已丢失（可恢复），否则是 disk store 终态故障（`dfkv_server_healthy == 0`，
需重启或替换）。逐轨定位用
`dfkv_server_ib_device_healthy{device}`。rail 相关查询在 TCP-only server 或未启动
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
| initialized/resolved | startup log `initialized=N`; `dfkv_rdma_rails_initialized`; `dfkv_rdma_recv_segment_registered_rails` | 完成 anchor 和全部启动 MR 的固定 topology，auto 含选出的 rail |
| startup-ACTIVE | startup log `ACTIVE=N` / `inactive=...` | 仅启动瞬间快照 |
| current healthy | `dfkv_server_rdma_rails_active`; `sum(dfkv_server_ib_device_healthy)` | 当前 ACTIVE+LinkUp 数；至少 1 即可 placement |

仅在 **RDMA-enabled binary 实际启动 RDMA listener** 时额外输出（折叠进同一
`/metrics`）；TCP-only binary 或未启动 RDMA listener 时下列 family 不存在：

| 指标 | 类型 | 含义 |
|---|---|---|
| `dfkv_server_ring_eligible` | gauge | 见 §3.1（所有节点都输出）。rail 侧条件：至少一条通过门控的 initialized healthy rail；最后一条丢失立即为 0，0→非 0 需 recovery streak |
| `dfkv_server_rdma_rails_configured` / `dfkv_rdma_rails_initialized` / `dfkv_server_rdma_rails_active` | gauge | health monitor 固定 resolved 数 / 成功 anchor 初始化数 / 当前健康数；auto 模式均包含选出的 rail，无 `device` label |
| `dfkv_server_ib_device_healthy{device}` | gauge | 每个固定 initialized device 当前是否 `query_ok && ACTIVE && LinkUp`；部分为 0 不再等价于整节点退环 |
| `dfkv_rdma_completions_total` / `dfkv_rdma_completion_errors_total` | counter | RDMA 请求完成 / 错误完成 |
| `dfkv_rdma_active_conns` | gauge | 当前服务中的 RDMA 连接；连接关闭时，要等它占用的 endpoint、接收段和连接计量都释放之后才减 |
| `dfkv_rdma_v2_conns_opened_total` | counter | server 累计打开的 v2 连接 |
| `dfkv_rdma_pending_handshakes` | gauge | 已 accept、bootstrap 帧尚未收齐的连接数。正常接近 0；集中重连时会短暂升高（客户端建 QP 期间一直计入）；持续偏高 = 有只连接不发帧的对端（端口扫描、只做 TCP connect 的探活、卡住的客户端） |
| `dfkv_rdma_rejected_connections_total` | counter | 达到 `DFKV_RDMA_MAX_PENDING_HANDSHAKES` 后被直接关闭的 bootstrap 连接；增长时先看上一行是谁占住了名额。名额从 accept 占到客户端建完 QP 并发出帧为止，所以在大批客户端集中重连期间增长说明上限偏小，应调大 |
| `dfkv_rdma_accept_failures_total` | counter | bootstrap 监听 `accept()` 因 fd 或内存耗尽（EMFILE/ENFILE/ENOBUFS/ENOMEM 等非单连接错误）失败的次数；每次失败后退避 100 ms 再试。增长 = 进程 fd 上限或主机内存不足 |
| `dfkv_rdma_thread_start_failures_total` | counter | 因线程创建失败而拒绝的 bootstrap 连接（进程继续运行）。增长 = 触到线程数上限（systemd `TasksMax`、cgroup `pids.max`）或内存不足 |
| `dfkv_rdma_v2_put_writes_total` | counter | server 实际收到的 `WRITE_WITH_IMM` PUT |
| `dfkv_rdma_leaseput_oversize_rejects_total` | counter | 对象大于 `--max-msg` 而被答 `kInvalid` 的 leased PUT（client 的 `DFKV_RDMA_MAX_BLOCK_BYTES` 高于 server 上限）。增长时调大 `--max-msg` 或调小 client 上限 |
| `dfkv_rdma_recv_segment_bytes` / `max_bytes` / `chunks` | gauge | receive-pool 当前提交量 / hard budget / 已提交 chunk 数 |
| `dfkv_rdma_recv_segment_used_bytes` / `free_bytes` | gauge | 已提交 chunk 中所有 lease 占用 / 全部空闲字节，**含 staging 专用 chunk**；后者不等于连接可分配字节 |
| `dfkv_rdma_recv_segment_largest_free_range_bytes` | gauge | 任一 chunk 最大连续 free range，含 staging 专用 chunk |
| `dfkv_rdma_recv_segment_connection_free_bytes` / `connection_largest_free_range_bytes` | gauge | 排除 staging 专用 chunk 后，连接可用的全部 / 单块最大连续空闲字节；排查 bootstrap 连接拒绝时看此组 |
| `dfkv_rdma_recv_segment_staging_reserved_bytes` / `staging_reserved_free_bytes` | gauge | 初始 chunk 专供动态 GET 和 leased PUT 暂存的总额 / 当前空闲；已计入上述 hard budget，并非新增预算 |
| `dfkv_rdma_recv_segment_growths_total` / `growth_failures_total` | counter | 启动后 chunk 增长成功 / 因预算或分配失败 |
| `dfkv_rdma_recv_segment_shrinks_total` / `released_bytes_total` / `chunk_idle_ms` | counter / gauge | 空闲缩容次数 / 已返还字节 / 非初始chunk空闲保留期 |
| `dfkv_rdma_recv_segment_allocation_failures_total` | counter | grow 后仍无法满足的最终 allocation |
| `dfkv_rdma_pull_connections` | gauge | 当前 dynamic pull 连接数，包含新握手和 v2.28 dynamic 过渡连接 |
| `dfkv_rdma_pull_zerocopy_served_total` | counter | 为 RAM arena 命中准备的 zero-copy dynamic-pull READY 数，不等于客户端成功完成的 READ 数 |
| `dfkv_rdma_pull_zerocopy_register_failures_total` | counter | RAM arena 命中但逐次 MR 注册失败、退回拷贝路径的次数；持续增长通常是 memlock 上限或设备 MR 数量上限不够 |
| `dfkv_rdma_dynamic_get_active` / `dfkv_rdma_dynamic_get_bytes_active` | gauge | 当前持有的 GET staging 租约数 / 字节（staging held，不是在途 grant 数）：从授权到 RELEASE；client 从 RAM arena 读的 GET 在发 READY 时就归还 staging，此后不再计入 |
| `dfkv_rdma_dynamic_get_mr_active` | gauge | 当前进程存活的 exact dynamic READ MR；PullRelease 先撤销 MR，再释放 RAM pin，断连由 QP 销毁完成 fencing |
| `dfkv_rdma_server_grant_register_seconds` | summary（`_sum` / `_count`） | GET 授权为 client 的 READ 授权所花的时间：逐次注册的 MR，或开了 `DFKV_RDMA_SERVER_MW` 时的窗口绑定。`_sum / _count` 是每个授权的平均值 |
| `dfkv_rdma_server_mw_mode{dev}` | gauge | 这条 rail 的 GET 授权是否用内存窗口（`DFKV_RDMA_SERVER_MW` 开、且网卡支持 type-2B 窗口为 1；否则 0，用逐次 MR） |
| `dfkv_rdma_server_mw_binds_total` / `dfkv_rdma_server_mw_invalidations_total` | counter | 用窗口授权的 GET / RELEASE 时作废的窗口 |
| `dfkv_rdma_server_mw_invalidation_failures_total` | counter | 没能投递或以错误完成的作废；连接随之关闭，范围照样先隔离再复用 |
| `dfkv_rdma_server_mw_quarantine_bytes` | gauge | 已作废、正在隔离期内、还不能复用的 GET 范围（arena pin）字节数；不含尚未撤销的 bind 预留。实际隔离 + 授权时原子的 grant 预留共同受 `DFKV_RDMA_SERVER_MW_QUARANTINE_BYTES` 限制，预留转隔离或取消只扣一次 |
| `dfkv_rdma_server_mw_quarantined_total` | counter | 进入隔离期的 GET 范围（RELEASE 作废，或连接关闭时） |
| `dfkv_rdma_server_mw_quarantine_fallbacks_total` | counter | 因隔离区已满、或所需隔离期超过 `DFKV_RDMA_SERVER_MW_QUARANTINE_MAX_MS` 而改用逐次 MR 的 GET；持续增长说明这条 rail 上同时在途的 GET 字节太多，或上限偏小 |
| `dfkv_rdma_server_mw_fallback_grants_total` | counter | 有窗口的连接上改用逐次 MR 的 RAM 层 GET：所在 arena extent 还没为窗口注册好（注册在后台，第一次用到时开始）、注册失败，或超出 `DFKV_RDMA_SERVER_MW_ARENA_BYTES`。要从盘暂存的 GET 不用窗口，也不计入 |
| `dfkv_rdma_server_mw_fallback_connections_total` | counter | 分配不到内存窗口（`ibv_alloc_mw` 失败，例如超过网卡的 `max_mw`）、整条连接改用逐次 MR 的连接 |
| `dfkv_rdma_server_mw_region_registrations_total` / `dfkv_rdma_server_mw_region_register_failures_total` | counter | 为窗口注册的 arena extent（每条 rail 各一次）/ 注册失败（10 s 后才会再试） |
| `dfkv_rdma_server_mw_arena_registered_bytes{dev}` | gauge | 这条 rail 上为窗口注册的 arena 字节数 |
| `dfkv_op_latency_seconds{op="get"}`（arena pull） | histogram | 记录 RAM handler 准备耗时，与 staged range handler 的计时边界一致；成功 PullRelease 后才提交样本，不包含客户端持有 grant、RDMA READ 或 MR 撤销时间。端到端延迟看客户端 bench，不能用此序列替代 |
| `dfkv_rdma_connection_bytes{class=\"data|control\"}` | gauge | data/control connection 当前 lease 字节；应随 adaptive class 而非 logical max 增长 |
| `dfkv_rdma_recv_segment_registered_rails` | gauge | 启动时成功打开（anchor）并通过试注册（初始 chunk 注册一次随即释放）的 rail 数；receive chunk 不再常驻整块注册，每条连接只注册自己的 slots |
| `dfkv_rdma_v2_ready` | gauge | 初始 receive chunk 与 rail anchor 是否就绪 |
| `dfkv_rdma_segment_evictions_total` / `dfkv_rdma_idle_reclaims_total` | counter | hard budget 压力淘汰 / idle timeout 回收连接 |
| `dfkv_rdma_unstaged_misses_total` | counter | 不分配 staging、直接回 kNotFound 的 GET（先查索引确认不存在）；以前 miss 也先租一段 staging，pool 紧张时还要等内存或驱逐别的连接 |
| `dfkv_rdma_pressure_unsatisfiable_total` | counter | receive pool 不足时没有驱逐任何连接就放弃的请求：没有一条空闲连接的租约释放后能放下它（不同 rail 的 chunk、释放后连同相邻空闲区间也不够大），或请求本身超过整个预算 |
| `dfkv_rdma_sessions_kept_total` / `dfkv_rdma_session_peer_closed_total` | counter | 保留了会话通道的连接（`DFKV_RDMA_SESSION_FD`，client 请求且 server 开启）/ 因 client 关闭会话通道而立即结束的连接 |
| `dfkv_rdma_stop_drains_cut_total` | counter | 停止时 `DFKV_RDMA_STOP_DRAIN_MS` 到期、仍有连接持有授权而被关掉的次数 |
| `dfkv_rdma_stop_grants_refused_total` | counter | 停止排空期间拒绝的新授权：GET 答 miss、leased PUT 答 `kCacheFull` |
| `dfkv_rdma_grant_expiries_total` | counter | 持有授权（GET 的 READY、PUT 租约、多窗口 PUT 的下一窗）而静默超过 `DFKV_RDMA_GRANT_TIMEOUT_MS` 被关掉的连接；增长 = client 崩溃或超时放弃，日志里有对端地址 |
| `dfkv_uring_reads_total` / `dfkv_uring_init_fallbacks_total` | counter | io_uring 路径实际读 / 初始化失败回退同步连接数 |
| `dfkv_rdma_rail_active_conns{dev}` | gauge | 每个本地 HCA 上当前连接数 |
| `dfkv_rdma_rail_completions_total{dev}` / `dfkv_rdma_rail_completion_errors_total{dev}` | counter | 每 rail 请求完成 / 错误完成 |
| `dfkv_rdma_rail_retry_exhausted_peers{dev}` | gauge | 最近 30 s 内在这个设备上有连接以 RETRY_EXC 或 RNR_RETRY_EXC 结束的不同对端 IP 地址数（按地址计，不按 QP；同一台机器上不同 pod 的 client 各算一个；每 rail 最多记 64 个）。只有一个通常是那个 client 自己（进程被杀、卡住）；同一设备上有好几个、而本节点别的设备干净，多半是这个设备的端口或链路问题。只用于观察和告警（`DfkvRdmaRailRetriesExhaustedForManyPeers`：5 分钟内最大值 ≥3 即触发，因为 client 在一次突发之后很快就不再用这个设备，数值随之回落），不影响放置 |
| `dfkv_rdma_rail_wc_errors_total{dev,status}` | counter | 上一行错误完成按 work completion status 分解；每条连接在首个失败完成处终止，所以同一 `dev` 各 `status` 之和等于 `dfkv_rdma_rail_completion_errors_total{dev}`。series 在首次出现时才导出，告警用上一行的总量；对端地址不做 label，见 §3.7 |
| `dfkv_rdma_rail_put_writes_total{dev}` / `dfkv_rdma_rail_put_bytes_total{dev}` | counter | 每 rail 收到的 PUT one-sided writes / payload bytes |

RAM zero-copy pull 的 cache-hit、read-bytes 和采样 GET 延迟在有效 PullRelease
完成时计入。准备阶段探测和失败后 staged fallback 不重复计数；断连中止不当作成功读取。

v2.31 删除 server/client GET-WRITE、固定 pull MR/MW、legacy connection 和
ambiguous GET quarantine 的专属序列；不保留值为零的兼容别名。GET 传输看
client pull READ/release 与 server exact-grant/pin 指标。

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
| `dfkv_read_coalesce_held_reads_total` | counter | follower 自己读、不等 leader 的次数：它所在的 serve 线程手里还有自己登记、尚未完成的读（等下去可能和别的线程互等，例如两条连接以相反顺序读同一批 key）。非零只说明合并少了一次，不是故障 |
| `dfkv_put_key_lock_waits_total`, `dfkv_put_key_lock_wait_us_total` | counter | PUT 等同一个 key 的另一个 PUT（RAM write-back 下 PUT 按 key 串行）的次数和累计微秒；不同 key 不互等 |

slab 引擎内部（**resolved engine=slab 时输出**——按运行时实际引擎判定，不设 flag/env 的默认也命中；file 引擎无此系列）：
| 指标 | 类型 | 含义 |
|------|------|------|
| `dfkv_slab_put_busy_total` | counter | 盘层以 `kCacheFull` 拒绝、但不是因为缺槽的 PUT：该 key 仍占着一个未结束操作的槽，或它的延迟删除等在途读写超过 5 秒上限。节点健康，客户端算作未存、不冷却节点；2.39 及更早版本这里返回 `kIOError`。不计入 `dfkv_slab_put_no_slot_total`。持续升高说明有 pin 或槽一直不释放 |
| `dfkv_retain_pending_applied_total` | counter | 挂起的 RETAIN SET（值还只在 RAM、尚未落盘时收到的 SET，客户端得到 `kProtectedNew`）在值落盘时被应用的次数 |
| `dfkv_retain_pending_denied_total` | counter | 其中盘在应用时拒绝的次数（留存预算已满，`kDenied`）：客户端被告知受保护，实际没有保护。持续增长说明留存预算不够 |
| `dfkv_retain_pending_entries` | gauge | 当前挂起、等值落盘的 SET 数。值落盘、被丢弃或被删除时移出；上限 2^20，满后对只在 RAM 的值的 SET 一律 `kDenied` |
| `dfkv_slab_put_no_slot_total` | counter | 盘层因分配不到槽而以 `kCacheFull` 拒绝的写入：既有客户端 PUT，也有 RAM 层向盘刷写的每一次尝试（重试各算一次）。分配不到槽指该 size class 的槽全部被在途读写 pin 住或被留存（`DFKV_RETAIN_*`）保护，也没有可以挪用的 extent；分配器极少见的索引插入失败也计入这里。节点是健康的，客户端把被拒的 PUT 算作未存、不冷却节点；2.38 及更早版本这里返回 `kIOError`，会被客户端当成节点故障。持续升高说明留存预算或并发占满了某个 size class |
| `dfkv_slab_dio_write_fallback_total` / `dfkv_slab_dio_read_fallback_total` | counter | direct 模式下回退 buffered 的写/读——**非零升高 = page cache 悄悄回来了**（对齐条件被破坏），direct 部署重点盯 |
| `dfkv_slab_table_sync_total` | counter | slots.tbl 的周期同步与关闭时同步次数（`DFKV_SLAB_TABLE_SYNC_MS`；默认 `boot` 下 100 ms、`auto` 下 5 s 一次，`off` 下只在关闭时同步，所以 `off` 运行期为 0 是正常的） |
| `dfkv_slab_extent_steals_total` / `dfkv_slab_extent_returns_total` | counter | 跨 class extent 抢占（伴随驱逐，容量失衡信号）/ 全空 extent 主动回池（无损再平衡） |
| `dfkv_slab_cold_steals_total` | counter | 命中"全局最冷 extent"护栏的跨 class 抢占（donor 全部内容都冷于 `DFKV_SLAB_COLD_STEAL_WINDOW` 窗口才走此路，满环自噬护栏；0=窗口关闭） |
| `dfkv_slab_watermark_evictions_total` / `dfkv_slab_watermark_slot_removals_total` | counter | 全局水位回收移除的对象数，包含最后一个 resident 经整 extent 回调清除的对象；不再以 extent 为单位 |
| `dfkv_slab_watermark_extent_clears_total` | counter | 成功的整 extent 元数据清除写入数，不等于对象数或 fdatasync 次数 |
| `dfkv_slab_watermark_ticks_total` / `dfkv_slab_watermark_active` | counter / gauge | bounded eviction tick 数 / hysteresis 是否仍在 drain |
| `dfkv_slab_watermark_last_tick_us` / `max_tick_us` | gauge | 最近 / 进程最大 watermark 锁区间微秒数 |
| `dfkv_slab_watermark_max_victims_per_tick` / `dfkv_slab_watermark_max_visits_per_tick` | gauge | 每盘每 tick 的 slot 移除上限 / 扫描上限；替代旧 max-extents gauge |
| `dfkv_slab_watermark_slot_record_clears_total` / `dfkv_slab_watermark_record_clear_bytes_total` | counter | 全局回收成功的单 slot tombstone 写入数 / 单 slot 与整 extent 元数据写入字节；不代表同步持久化或物理写放大 |
| `dfkv_slab_watermark_freed_slot_bytes_total` / `dfkv_slab_watermark_scan_visits_total` | counter | 全局回收释放的分配字节 / 扫描位置次数 |
| `dfkv_slab_disk_run_epoch` / `dfkv_slab_disk_capacity_bytes` | gauge | 按 `disk`、`disk_index` 标记的运行 epoch / 容量；各盘引擎不同时只列 slab 盘 |
| `dfkv_slab_disk_watermark_high_bytes` / `dfkv_slab_disk_watermark_low_bytes` / `dfkv_slab_disk_reclaim_interval_ms` / `dfkv_slab_disk_watermark_active` | gauge | 每盘解析后的水位、回收周期及活跃状态 |
| `dfkv_slab_disk_watermark_high_observations_total` | counter | 每盘回收线程实际观察到 used > high 的次数；不等于回收次数 |
| `dfkv_slab_disk_watermark_last_high_bytes` / `dfkv_slab_disk_watermark_peak_high_bytes` | gauge | 最近 / 本次运行最大超高水位观察值；历史 peak 不能单独证明新阶段跨水位 |
| `dfkv_slab_table_rebuilt_objects` / `dfkv_slab_rebuild_scanned_bytes` / `dfkv_slab_rebuild_scan_chunks` / `dfkv_slab_rebuild_sparse_ranges` / `dfkv_slab_rebuild_mmap_scans` | gauge | 上次启动冷升从 slots.tbl 恢复的对象数 / 扫描字节 / chunk / sparse range / mmap 的表数（冷升目录规模诊断） |
| `dfkv_slab_rebuild_corrupt_records_total` / `dfkv_slab_rebuild_rejected_records_total` / `dfkv_slab_rebuild_sequential_fallbacks_total` | counter | 启动 rebuild 丢弃的损坏记录 / 因几何不安全被清除的合法记录 / sparse seek 退化为顺序扫描的次数（任一非零都值得看日志） |
| `dfkv_slab_deferred_removes_total` | counter | 被在飞 I/O 延迟执行的 Remove |
| `dfkv_slab_inflight_keys` / `dfkv_slab_prep_holds` | gauge | 锁外 I/O 在飞 key 数 / 未释放的异步 prep 持有数（持续增长 = 泄漏） |
| `dfkv_slab_reclaimed_total` | counter | 后台回收线程预驱逐的 slot 数（`DFKV_SLAB_RECLAIM_MS`，默认 50ms）——持续为 0 且 PUT 延迟高 = 回收被关或池未满，先查 `--slab-reclaim-ms` |
| `dfkv_slab_rebalanced_total` | counter | 回收线程从冷 class 搬给热 class 的 extent 数（类再平衡）——换模型/尺寸迁移期应看到增长，稳态应静止 |
| `dfkv_slab_batched_writes_total` / `dfkv_slab_uring_write_batches_total` | counter | 进入 batched store visit 的 payload 写数 / io_uring 单次 submit 轮数 |
| `dfkv_slab_bind_wipes_total` | counter | extent 绑定新 class 前完成的旧 slot-grid 清理 |
| `dfkv_slab_payload_crc_failures_total` | counter | 整值读取时 payload 与写入时记录的 CRC32C 不符的次数；该值按 miss 返回并像 REMOVE 一样删除。旧版本写的记录不带 CRC，不检查 |
| `dfkv_slab_metadata_io_errors_total` / `dfkv_slab_unclean_resets_total` | counter | slots.tbl/epoch 持久化失败 / 冷升时丢弃 dirty epoch |
| `dfkv_slab_warm_restarts_total` | counter | 非干净退出后保留了缓存的盘数（`DFKV_SLAB_WARM_RESTART` 为 `auto` 或 `boot` 时才会非零）；与 `unclean_resets` 互斥，清零的原因见启动日志 |
| `dfkv_slab_trust_tokens_held` | gauge | 当前持有可信令牌的盘数。正常运行时等于盘数（同步表的瞬间会短暂变少）；长期偏少说明令牌建不出来或该盘出过元数据错误，此时非干净退出会清零 |
| `dfkv_slab_eviction_record_clears_total` / `dfkv_slab_record_writes_total` | counter | allocator 复用前持久清除记录 / 已提交 slots.tbl record 写数 |
| `dfkv_slab_capacity_bytes` / `dfkv_slab_allocated_bytes` / `dfkv_slab_payload_bytes` / `dfkv_slab_internal_fragmentation_bytes` | gauge | 物理 payload 容量 / 已占 slot / 逻辑 payload / 内部碎片 |
| `dfkv_slab_allocator_objects` / `dfkv_slab_committed_objects` | gauge | allocator resident（含未提交写）/ reader-visible 已提交对象 |
| `dfkv_slab_classes` / `dfkv_slab_bound_extents` / `dfkv_slab_pool_extents` | gauge | live class 数 / 已绑定 extent / 未绑定共享池 extent |
| `dfkv_slab_class_extents{slot_size}` / `dfkv_slab_class_resident_objects{slot_size}` / `dfkv_slab_class_free_slots{slot_size}` | gauge | 每个 live size class 的 extent、resident、可立即分配 slot |
| `dfkv_slab_class_allocated_bytes{slot_size}` / `dfkv_slab_class_useful_bytes{slot_size}` / `dfkv_slab_class_fragmentation_bytes{slot_size}` | gauge | 每 class 的占用、有效 payload、内部碎片 |
| `dfkv_slab_class_read_heat{slot_size}` / `dfkv_slab_class_put_total{slot_size}` | gauge / counter | 每 class 的衰减读热度 / 累计 PUT |
| `dfkv_slab_class_tail_age{slot_size}` | gauge | 该 class 下一个要淘汰的值自写入或最近一次读以来经过的分配器操作数（多盘取最大）；各 class 差得多说明空间分配与当前负载不匹配（`DFKV_SLAB_BALANCE`） |
| `dfkv_slab_balance_moves_total` / `dfkv_slab_balance_max_hold_us` | counter / gauge | `DFKV_SLAB_BALANCE=1` 时按 tail age 在 class 间挪动的 extent 数 / 一次挪动占用 store 锁的最长时间 |
| `dfkv_slab_failed_disks` / `dfkv_slab_healthy` | gauge | fail-closed disk 数 / 全部 slab disk 是否健康 |

PUT 准入门（**仅 `--put-inflight-limit > 0` 时输出**）：
| `dfkv_put_busy_total` | counter | 被准入门以 kCacheFull 快速拒绝的 PUT（受控 miss，替代深队列尾延迟） |

RAM 热层（**仅 `DFKV_RAM_TIER=1` 时输出**；关时无此系列，向后兼容）：
| 指标 | 类型 | 含义 |
|------|------|------|
| `dfkv_ram_hit_total` / `dfkv_ram_miss_total` | counter | GET 命中 RAM / 未命中落盘（命中率 = hit/(hit+miss)） |
| `dfkv_ram_put_total` | counter | 写直通进 RAM 的 PUT 数 |
| `dfkv_ram_put_bypass_total` | counter | **背压**：arena 满（flush 落后）→ PUT 旁路直写盘，非零即 flush 跟不上 |
| `dfkv_ram_parked_puts` | gauge | 正在等自己这个 key 的属主值的 PUT 数：disk-ACK 下每个被接纳的 PUT（等刷盘）；RAM-ACK 下只有等首写者的值可见的重复 PUT（通常是瞬间）。disk-ACK 下持续偏高说明刷盘跟不上 |
| `dfkv_ram_put_wait_timeouts_total` | counter | 等待时间用完（从所在请求批次到达算起 `DFKV_RAM_PUT_WAIT_MS`）后回 `kCacheFull` 的 PUT 数，停止期间到时的也算在这里；值本身仍在 RAM 里继续刷盘 |
| `dfkv_ram_put_waits_canceled_total` | counter | 停止时被提前回 `kCacheFull` 的等待中 PUT 数（含停止开始后本来需要等待的 PUT） |
| `dfkv_ram_ack_backpressure_total` | counter | 正的 RAM-ACK dirty 水位触发的新 key 接纳拒绝；水位为 0 时计数强制等待落盘的请求。已有 key 先解析 owner，不因水位切换覆盖首写者 |
| `dfkv_ram_ack_backpressure_waits_total` | counter | 在水位处等待过的 RAM-ACK 新 key PUT 数（`DFKV_RAM_ACK_BACKPRESSURE_WAIT_MS`） |
| `dfkv_ram_ack_backpressure_wait_timeouts_total` | counter | 等待结束时仍被水位拒绝（回 `kCacheFull`）的 PUT 数；也计入 `dfkv_ram_ack_backpressure_total` |
| `dfkv_ram_ack_backpressure_wait_microseconds_total` | counter | PUT 在水位处等待的总时长（微秒），除以 waits 得平均等待 |
| `dfkv_ram_promoted_total` | counter | 整值冷读晋升：优先让 O_DIRECT 直接读入隐藏 arena reservation，成功后以 born-durable 身份发布；staged fallback 则读后复制晋升。不进 flushq、零重复刷盘、随时可驱逐；健康冷→热运行中首次读增长，随后应转为 `dfkv_ram_hit_total` 增长且磁盘字节不再增加 |
| `dfkv_ram_promote_declined_total` | counter | RAM 没有接收的晋升（没有空间，或这类值不进 RAM） |
| `dfkv_ram_promote_deferred_total` | counter | 仅 `DFKV_RAM_PROMOTE=second` 时输出：因为这个 key 在 `DFKV_RAM_PROMOTE_WINDOW_MS` 内没被读过而没有装进 RAM 的整值冷读。与 `dfkv_ram_promoted_total` 之比高 = 读的多半是只读一次的数据（扫描）；热数据的命中率却不涨，说明窗口比复用间隔短 |
| `dfkv_ram_recent_reads` | gauge | 仅 `DFKV_RAM_PROMOTE=second` 时输出：当前记着"窗口内读过"的 key 数（上限 262144） |
| `dfkv_ram_reservations_aborted_total` | counter | 晋升 reservation 没有装成副本就结束（读失败，或被 REMOVE、磁盘上的新值抢先） |
| `dfkv_ram_evicted_unread_total` | counter | 因空间被淘汰、进 RAM 后从未被读过的值（读 = 从 RAM 送出过数据：TCP 读、RDMA 读和 RDMA pull 都算，EXIST 不算）；与 `dfkv_ram_evictions_total` 之比高说明 RAM 里放的多半没人读 |
| `dfkv_ram_evicted_age_ms_total` | counter | 被淘汰的值在 RAM 里驻留时长之和（毫秒），除以 `dfkv_ram_evictions_total` 得平均驻留时长 |
| `dfkv_ram_flushed_total` / `dfkv_ram_flush_dropped_total` | counter | RAM slot 落盘转 DURABLE / flush 全部尝试失败（或目标盘 store 已终态故障）后丢弃 |
| `dfkv_ram_flush_retries_total` | counter | 失败后还会再试的 flush 尝试次数。已 RAM-ACK 的值按 50 ms、200 ms、800 ms、3.2 s 间隔重试（共 5 次尝试，约 4.25 s），期间对象仍可读、仍计入 dirty；仍在等落盘结果的 PUT（disk-ACK）不加间隔。增长而 `dfkv_ram_flush_dropped_total` 不涨 = 盘侧有瞬时写失败但没有丢数据 |
| `dfkv_ram_flush_no_room_total` | counter | 盘回 `kCacheFull`（健康但此刻没有可用的槽）的 flush 尝试次数。不算失败：不计入 `dfkv_ram_flush_retries_total`，不影响 `dfkv_ram_healthy`；已 RAM-ACK 的值留在 RAM 里按 50 ms→3.2 s 的间隔继续试，最长 60 秒；停止排空期间改为不加间隔再试 5 次（`flush_retries`） |
| `dfkv_ram_flush_refused_total` | counter | 盘明确表示永远不会接收而丢弃的 RAM 值：`kInvalid`（值大于盘的 extent）或 `kQuotaExceeded`。不重试、不影响 `dfkv_ram_healthy`、不计入 `dfkv_ram_post_ack_flush_failures_total`；等落盘结果的 PUT 得到盘的这个回答。正常情况下为 0：大于盘上限的值本来就不进 RAM（直接得到 `kInvalid`），有配额的租户也不走 RAM 写回 |
| `dfkv_ram_flush_no_room_dropped_total` | counter | 因盘没有槽而丢弃的值：disk-ACK 下等落盘的 PUT 得到 `kCacheFull` 的值、60 秒内一直没等到槽的已确认值，以及停止排空期间不加间隔再试 5 次仍没有槽的已确认值。后者之后的 GET 会 miss，相当于被淘汰；不计入 `dfkv_ram_post_ack_flush_failures_total` |
| `dfkv_ram_ack_total` | counter | 在落盘前由 RAM 确认的 PUT 数。同一 key 的重复 PUT 在其值已确认、仍在等落盘时也立即得到确认并计入这里，所以它数的是 PUT，不是值；丢失的值看下一行（按值计，每个值一次） |
| `dfkv_ram_post_ack_flush_failures_total` | counter | **已向 client 确认（RAM-ACK）、随后因落盘失败被丢弃的值**（盘没有槽而丢弃的不在此列，见 `dfkv_ram_flush_no_room_dropped_total`），即已确认但丢失的数据；每次丢弃另有限速 WARN `ram tier: dropped N object(s) ...`。告警 `DfkvRamAcknowledgedDataLost` |
| `dfkv_ram_healthy` | gauge | 发生丢弃后为 0，直到距最后一次丢弃至少 30 秒且连续 64 次 flush 成功（中间没有任何失败尝试）才恢复为 1；0 时 `/healthz`、`/readyz` 为 503。不再是“直到重启” |
| `dfkv_ram_flush_threads` | gauge | shard 最小值调整后的实际 flusher 数（不是原始请求值） |
| `dfkv_ram_evictions_total` | counter | RAM slot 容量压力淘汰数（含内联与后台回收两路） |
| `dfkv_ram_reclaimed_total` | counter | 其中由后台回收线程预驱逐的数量（`DFKV_RAM_RECLAIM_MS`，默认 10ms；flush 积压 >4096 时自动歇拍，此计数暂停属预期） |
| `dfkv_ram_rebalanced_total` | counter | RAM 层类再平衡搬动的 extent 数（增长阶段不受 flush 积压歇拍影响——从冷 donor 搬 durable extent 恰是 flush-gated 时唯一能扩收速的动作） |
| `dfkv_ram_objects` / `dfkv_ram_flush_backlog` | gauge | 当前 RAM 常驻块 / 待 flush（含已出队、正在 flush，均尚未 DURABLE）的块数 |
| `dfkv_ram_budget_bytes` / `dfkv_ram_arena_bytes` / `dfkv_ram_large_budget_bytes` | gauge | RAM 层总预算（arena+dedicated）/ arena 固定预留 / 总预算内大对象 dedicated 预留 |
| `dfkv_ram_arena_locked` | gauge | arena 是否锁在内存里（1/0）；为 0 时开了 swap 的主机在内存紧张时可能把 arena 换出 |
| `dfkv_ram_arena_hugepage_bytes` | gauge | 预触页结束时 arena 里由透明大页承载的字节数，启动时取一次样，之后不更新（内核后台可能把剩下的 4K 页再合并成大页，这里不会反映）。接近 `dfkv_ram_arena_bytes` 表示基本都是大页（arena 不是 2 MiB 整数倍时尾部不足一个大页的部分是 4K 页）。为 0 有两种情况：申请了但没拿到（系统设置为 `never`）；或者没有申请、也就没有测量（`DFKV_RAM_TIER_HUGEPAGES=off`，或 arena 小于 2 MiB），这时系统设置若是 `always`，arena 实际仍在大页上 |
| `dfkv_ram_used_bytes` / `dfkv_ram_large_used_bytes` | gauge | 已用（常驻 slot + dedicated 分配）/ 其中 dedicated 大对象已用——与上组预算行做水位比 |
| `dfkv_ram_payload_bytes` | gauge | 常驻值的有效载荷字节数（arena 与 dedicated 合计）。`dfkv_ram_used_bytes` 减去它是 slot 取整的损失；减去 `dfkv_ram_large_used_bytes` 后与 `dfkv_ram_arena_bytes` 的比值是 arena 的实际装载率，大对象时可能远低于 used/arena（见 DEPLOY.md“大对象为主时调大 extent”） |

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
| `dfkv_mds_group_ring_eligible_nodes{group}` / `dfkv_mds_group_degraded_nodes{group}` | 当前可参与 placement 的节点数 / 自报不可 placement（无健康 rail，或 disk store 终态故障）但 lease 仍在线的节点数；后者 >0 触发 `DfkvGroupMembersOutOfPlacement` |
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
| `dfkv_mds_deregister_requests_total` | counter | kDeregister：被 `dfkvctl drain` 排空的节点停止时删除自己的记录（DEPLOY §4c-1） |
| `dfkv_mds_list_requests_total` | counter | ListMembers 次数 |
| `dfkv_mds_lease_grants_total` | counter | etcd lease 授予：只在 member 记录不存在或其租约已过期时发生，稳态下不增长 |
| `dfkv_mds_record_puts_total` / `dfkv_mds_record_puts_skipped_total` | counter | 写进 etcd 的 member/client 记录 / 只续租约、没有写的心跳（记录未变，或只有统计变了且未到 `DFKV_MDS_STATS_PUT_MS`）。稳态下前者每节点约每分钟 1 次 |
| `dfkv_mds_list_cache_hits_total` | counter | 由 `DFKV_MDS_LIST_CACHE_MS` 缓存回答、没有读 etcd 的 member list 请求 |
| `dfkv_mds_client_list_cache_hits_total` | counter | 同上，client list（`kListClients`）请求 |
| `dfkv_mds_fenced_refusals_total` | counter | 被拒绝的已 fence 节点的注册/心跳（`DFKV_MDS_FENCING=1`） |
| `dfkv_mds_etcd_errors_total` | counter | etcd I/O 失败 |
| `dfkv_mds_max_conns` | gauge | 生效的并发连接上限：`DFKV_MDS_MAX_CONNS` 按进程 fd 上限折算后的值；小于配置值说明 `LimitNOFILE` 不够 |
| `dfkv_mds_accept_failures_total` | counter | MDS 监听 `accept()` 因 fd 或内存耗尽失败的次数；每次失败后退避 100 ms 再试。增长 = fd 上限不足，同时 MDS 连 etcd 也可能失败 |
| `dfkv_mds_thread_start_failures_total` | counter | 因 handler 线程创建失败而拒绝的连接（进程继续运行） |
| `dfkv_mds_members` | gauge | 上次 List 返回的成员数 |
| `dfkv_mds_client_register_requests_total` / `dfkv_mds_client_keepalives_total` / `dfkv_mds_client_list_requests_total` | counter | consumer 注册 / 心跳 / ListClients |
| `dfkv_mds_clients` | gauge | 上次 ListClients 返回的 consumer 数 |
| `dfkv_mds_etcd_requests_total{op}` / `dfkv_mds_etcd_request_errors_total{op}` | counter | etcd HTTP 请求 / 传输或 HTTP status 失败，`op=lease_grant|lease_keepalive|put|range|metrics_range|get|delete`（`get` 是心跳读单条记录及其租约，`delete` 是 fence 和 kDeregister 删除记录）；`metrics_range` 是每次 MDS scrape 同时用于 readiness 与 group 聚合的唯一 range，和业务控制面流量分开计数 |
| `dfkv_mds_etcd_request_duration_seconds{op}` | histogram | 同五类 etcd HTTP 请求延迟（50µs–60s + `+Inf`）；用于区分本地控制面排队、常规 etcd 慢响应和 scrape 自身的聚合 range |
| `dfkv_mds_local_member_leases` / `dfkv_mds_local_client_leases` | gauge | 本 MDS 进程当前缓存的 member/client lease shortcut 数；只是 etcd 权威状态的有界优化 |
| `dfkv_mds_local_member_leases_pruned_total` / `dfkv_mds_local_client_leases_pruned_total` | counter | 因多个 TTL 未在本 MDS 使用而丢弃的本地 shortcut；churn 下增长正常，下次心跳会 fresh grant+Put |
| `dfkv_mds_legacy_frames_total` | counter | 以 v1.x 旧控制面帧（42/10 字节帧）服务的请求数；仅在 `DFKV_MDS_ACCEPT_LEGACY=1` 的混合代际迁移期出现，**归零后应撤掉该 env** |
| `dfkv_mds_list_withheld_total` | counter | 被拒绝下发的 member list 次数（`ListMembers`/`ListTopology` 返回 `kIOError`）：etcd 视图里缺少“心跳已到本 MDS、但 etcd 没接受写入”的 member。**增长 = etcd 拒绝写入（典型为空间配额用尽）且成员 lease 已开始过期**，客户端此时停在 last-good ring；连续约 4 分钟触发告警 `DfkvMdsMemberListWithheld`。心跳在本 MDS 失败之后真正下线的节点也会让它增长，至多一个 `DFKV_MDS_UNWRITTEN_HOLD_MS` 窗口 |
| `dfkv_mds_etcd_write_failing_seconds` | gauge | etcd 拒绝或写失败（lease grant / keepalive / put）已持续的秒数：上次写成功之后第一次失败起算；写成功时为 0；最近 30 秒（成员 lease TTL）内没有任何写入尝试时也回到 0（没有心跳要写，就没有成员会因此过期），不会在空闲的 MDS 上一直增长。成员 lease 30 秒到期后就会从 etcd 消失，超过 10 秒触发 `DfkvMdsEtcdWritesRefused`。etcd 的回答（HTTP 状态和错误体，例如 `database space exceeded`）记在 MDS 日志 `mds: etcd <op> failed: ...`，每种操作 10 秒至多一行 |
| `dfkv_mds_member_decode_errors_total` | counter | 成员或客户端列表请求中遇到无法解码、被跳过的 etcd 记录的次数：每次列表请求对每条坏记录各计一次（反映暴露次数，不是不同坏记录的个数）；日志 `mds: skipped a record in etcd ...` 给出键名（30 秒至多一行），告警 `DfkvMdsMemberRecordsUndecodable` |
| `dfkv_mds_unwritten_members` | gauge | 本 MDS 上最近一次心跳没能写进 etcd 的 member 数（`DFKV_MDS_UNWRITTEN_HOLD_MS` 窗口内；只计 heartbeat，注册失败不计）。>0 而上一行不增长 = 写入失败但 etcd 仍列出这些 member（旧 lease 尚在或已由其他 MDS 写入），list 照常下发 |

### 3.3 客户端（SGLang 插件 `/metrics` 与原生 C 快照）
SGLang 后台线程每 `client_stats_poll_s`（默认 10s，0=关）读取一次 C 客户端快照，
把下列有界 family 镜像到插件 `/metrics` 并追加 `{tp_rank}`；request path 不做
Prometheus 调用。vLLM 使用同一 allowlist，区别只是在 family 前加 `vllm:`，见 §3.5。
原生客户端声明的每个 counter/gauge family 都在 allowlist 里，或在
`integration/common/tests/test_client_metrics.py` 的 `_NOT_MIRRORED` 里写明原因；
测试扫描原生源码中的 `# TYPE` 声明来检查。rail 与 peer 故障信号（
`dfkv_rdma_client_no_compatible_rail_total`、`dfkv_rdma_client_remote_rail_*`、
`dfkv_rdma_client_cross_rail_retr*`、`dfkv_rdma_client_admission_failures_total`、
`dfkv_rdma_client_pull_failures_total` 等）因此也出现在插件 `/metrics` 上。

| 指标 | 类型 | 含义 |
|---|---|---|
| `dfkv_client_stats_snapshot_success` / `dfkv_client_stats_snapshot_timestamp_seconds` / `dfkv_client_stats_snapshot_errors_total` | gauge / gauge / counter | 最新 poll 是否成功 / 最后成功 Unix 时间 / 失败累计；失败时保留 last-good 数据并显式置 success=0 |
| `dfkv_client_transport_info{transport}` | gauge | live transport identity（`rdma`/`tcp`），恒为 1 |
| `dfkv_client_ops_served_total` / `dfkv_client_io_errors_total` / `dfkv_client_health_checks_total` | counter | 收到节点应答（命中、未命中，以及 `cache full`、`quota exceeded`：凡是节点自己给出的状态都算应答，证明节点存活）/ 传输失败（不含传输层未拨号直接拒绝的请求，这类拒绝不计数、也不触发回避）/ endpoint health check |
| `dfkv_client_unhealthy_skips_total` / `dfkv_client_busy_suppressed_total` | counter | 被跳过的请求（peer 在回避期内，或回避期已过但试探请求尚未返回）/ busy 后主动抑制请求 |
| `dfkv_client_peer_marked_bad_total` / `dfkv_client_peer_recovered_total` | counter | peer 进入回避（每次事故或每次试探失败计一次；回避期内到达的失败只计入 `dfkv_client_io_errors_total`）/ 恢复（只在有请求得到应答时计；probe 应答不再计为恢复） |
| `dfkv_client_peer_errors_total{peer}` | counter | 逐 endpoint 错误；peer 集合在原生层硬限 4096 |
| `dfkv_read_scheduler_node_limit_waits_total` | counter | 空闲 read worker 被唤醒后，发现待执行的 shard 全部属于已达单节点上限的节点的次数。正常运行中也会出现（例如小环上对单个节点的大 batch）；若其他节点的读同时变慢且该计数持续高速增长，说明有慢节点 |
| `dfkv_client_ring_members` / `dfkv_client_mds_reachable` / `dfkv_client_mds_unreachable_polls_total` | gauge / gauge / counter | 当前环节点数 / 最近发现轮询是否可达 MDS / 不可达轮询累计 |
| `dfkv_client_ring_draining_members` | gauge | 环里正在排空的节点数（DEPLOY §4c-1）：不再写入，读在新 owner 未命中后才问它 |
| `dfkv_client_ring_absent_members` | gauge | 缺席成员数（DEPLOY §4c-2，`DFKV_RING_ABSENT_GRACE_MS`）：从视图消失、仍在宽限期内的成员，不再写入也不做普通读（只经 B 的 M/W 安全检查回读），REMOVE 和 RETAIN 也发给它 |
| `dfkv_client_ring_absent_dropped_total{reason}` | counter | 移出的缺席成员：`grace` 宽限期满，`bound` 超过 `max(1, N/8)` 的上限立即移出，`planned` 排空后下线、不保留 |
| `dfkv_client_ring_absent_removes_unconfirmed_total` / `_retried_total` / `_dropped_total` | counter | 缺席成员没确认的 REMOVE / 它回来后重发的次数（每次发送都计）/ 最终没确认的（超出队列上限、成员被移出、回来后一个宽限期仍未确认） |
| `dfkv_client_cooldown_reroutes_total{op}` | counter | 冷却改道（DEPLOY §4c-3）：owner 在本客户端回避期内、改走替补的 key（`get`、`exist`、`put`、`retain`）；`none`：owner 在回避期内但前几个候选都不能用，照旧跳过 |
| `dfkv_client_readback_reads_total{result}` | counter | 回读前任 owner（DEPLOY §4c-5）：`hit`、`miss`，以及被拒的候选：`refused_removed`（有节点说删过）、`refused_unknown`（有节点说不知道、没有记忆或问不到）、`refused_length`（本客户端见过长度不符）、`refused_gate`（登记门关着）、`refused_cooling`（前任 owner 在回避期）；`hit_rewritten`：删除后又在前任 owner 上重写、因此回读的次数（DEPLOY §4c-6，另外也计入 hit 或 miss） |
| `dfkv_client_readback_changes_refused_total{reason}` / `dfkv_client_readback_window_active` | counter / gauge | 没有回读窗口的环变化：`held`（被保护视图压得太久）、`chained`（窗口内又一次变化）/ 当前是否有窗口 |
| `dfkv_client_readback_memory_short_total{reason}` | counter | 记忆长期短于回读所需时长 H 的节点（`window`：`DFKV_REMOVE_MEMORY_MS` 小于 H）；每个节点计一次并打一条 WARNING |
| `dfkv_client_ring_reroute_episode_pending` | gauge | 回避期已结束、在替补上的 REMOVE 还没确认的 key 数；它和正在记账的 key 合计受 `DFKV_RING_REROUTE_KEYS` 限制，持续增长说明某个替补一直清不掉（例如它已经死了），改道写入快被拒绝了 |
| `dfkv_client_reroute_refused_total` | counter | 已记账的替补 key 达到 `DFKV_RING_REROUTE_KEYS` 后被拒绝的改道写入 |
| `dfkv_client_reroute_cleanup_total{result}` | counter | 回避期结束（或扇出目标回避期满）时对替补发的清理 REMOVE：`ok`，`queued`（失败，之后补发），`abandoned`（超过上限，或关闭前没发出去） |
| `dfkv_client_remove_fanout_total{result}` | counter | REMOVE 扇出的消息：`ok`，`error`，`queued_cooling`（目标在本客户端回避期内，记入清理队列） |
| `dfkv_client_remove_unconfirmed_total{reason}` | counter | 因为有目标没在这次调用里确认而返回失败的 REMOVE：`cooling`（目标在本客户端回避期内）、`failed` |
| `dfkv_client_registry_gate_open` / `dfkv_client_registry_unmarked_clients` | gauge | 登记门是否打开（冷却改道生效）/ 上次登记表里没带扇出标记（或标记里没有 `poll_ms`）的客户端数 |
| `dfkv_client_registry_gate_failures_total{reason}` | counter | 登记门由开到关的次数，按原因：`unmarked_client`、`stale_registry`、`no_registry`、`s_growing` |
| `dfkv_client_drain_requests_total{op,result}` | counter | 发给 key 的排空或缺席 owner 的第二次请求：`op=read`（新 owner 未命中后的读，只问排空 owner）、`remove`、`retain`（两边都发，含缺席 owner）；`result=served` 它有这个 key（或确认删除），`not_served` 没有或失败 |
| `dfkv_client_dedup_hits_total` / `dfkv_client_dedup_fetches_total` / `dfkv_client_dedup_wait_hits_total` / `dfkv_client_dedup_wait_timeouts_total` | counter | host rendezvous 命中、真实 fetch、等待后命中、等待后回退直连（等满 `DFKV_NODE_DEDUP_WAIT_MS`，或 leader 放弃 / 已发布条目不可用而提前回退，两者不区分） |
| `dfkv_client_gpu_dedup_hits_total` / `dfkv_client_gpu_dedup_fetches_total` / `dfkv_client_gpu_dedup_wait_hits_total` / `dfkv_client_gpu_dedup_wait_timeouts_total` | counter | 同上 CUDA IPC rendezvous。另：发布中的 slot 被收回（发布者超过 `DFKV_NODE_DEDUP_TAKEOVER_MS` 或该 key 被写入/删除）之后，到下一次 fetch 认领之前 slot 读作空，正在等的 follower 复查一次后放弃并自行读取，计入 wait_timeouts，也多一次远端读取 |
| `dfkv_client_dedup_length_mismatches_total` / `dfkv_client_gpu_dedup_length_mismatches_total` | counter | 会合查找发现同一个 key 已由同机另一个进程发布、但长度与本进程期望的不同（host 严格长度不等或超出缓冲，GPU 超出各段容量之和）的次数。该条目不被使用（本进程自行读取）；首次出现时打一条 WARN。dedup 的前提是同一个 key 在本机所有进程里是同样的字节，非零通常说明 key 漏了某个分片维度（例如分片的 payload 共用了一个 key），应关闭 dedup 并检查连接器的 key 坐标 |
| `dfkv_client_gpu_dedup_load_fences_total` / `dfkv_client_gpu_dedup_load_fence_failures_total` | counter | leader 在把一次 fetch 的结果发布到 GPU rendezvous 之前做的设备同步次数（每批一次）/ 设备同步失败、该批结果因此不发布的次数（follower 自行读取） |
| `dfkv_client_gpu_dedup_invalidate_stuck_total` | counter | 写入或删除一个 key 时的失效操作在某个 slot 上等了超过 `DFKV_NODE_DEDUP_TAKEOVER_MS` 的次数（每次失效每个 slot 最多记一次，同时打一条 WARN `gpu-dedup: invalidation, slot ...`，含 slot 序号、gen、state、alloc_seq）；或者一个只会以空 slot 结束的收回过程停在那里超过同一时间、失效不再等它的次数（同样打 WARN；正常进行中的收回会被等待，不计数）。前者说明持有者是同机更早版本的发布者，或是在已知的几条指令窗口里被杀的进程：失效会一直等下去，这次写入或删除不返回。非零就要查 WARN 和同机进程 |

原生 `KVClient` 的收敛操作指标（`op="put|get|exist|remove|retain"`）。retention 调用（`BatchRetain`）也计数：不带 TTL 的 REFRESH 是 ring 有 retention 时连接器的存在性探测，计入 `op="exist"`；SET 和带 TTL 的 REFRESH 计入 `op="retain"`，命中数是存在的 key 数：

| 指标 | 含义 |
|---|---|
| `dfkv_client_op_requests_total{op}` | 公开 scalar/batch/SG 调用次数；每次调用严格加 1 |
| `dfkv_client_op_keys_total{op}` | 调用提交的 key 数；TCP fan-out、RDMA pipeline、重试和 rendezvous 不重复计数 |
| `dfkv_client_op_hits_total{op}` | put/remove 确认成功数、get 命中数、exist 存在数；shm/CUDA rendezvous 命中也计入 |
| `dfkv_client_node_refusals_total{status}` | 节点拒绝写入的 key 数：`cache_full`（没有空间，或 RAM 层未落盘数据超过水位 `DFKV_RAM_ACK_HIGH_WATERMARK_PCT`），`quota_exceeded`（超出租户配额）。这些 key 没有写入，连接器按失败处理（不会重试），之后读取会 miss |
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
| `dfkv_rdma_client_v2_put_writes_total` | counter | 实际发出的 v2 one-sided PUT |
| `dfkv_rdma_client_unregistered_dest_registrations_total` | counter | GET 的目的地不在已注册内存（`dfkv_register_memory`）里时，为这次 GET 临时做的注册数；首尾相接或重叠的段合成一次。持续增长说明调用方没注册缓冲区：每次注册加注销约 40 µs，应改为启动时注册一次（日志里有限频 WARN） |
| `dfkv_rdma_client_unregistered_dest_span_fallbacks_total` | counter | 合并后的范围网卡不肯注册、退回逐段注册的次数 |
| `dfkv_rdma_client_unregistered_dest_memlock_fallbacks_total` | counter | 未注册的目的地一次整体注册被锁定内存上限（`ulimit -l`、`LimitMEMLOCK`）拒绝（ENOMEM/EAGAIN），改为每轮 READ 只注册这一轮的段的 GET 数 |
| `dfkv_rdma_client_mr_regions` / `dfkv_rdma_client_mr_registered_bytes` | gauge | 所有 active rail 已成功 anchor 后才发布的 host pool MR 区域数 / 声明字节数；任一 rail 失败时两者保持上次成功值 |
| `dfkv_rdma_client_mr_registration_rejections_total` | counter | 非法 range、rail anchor 或 `ibv_reg_mr` 失败而未发布（已回滚）的声明次数 |
| `dfkv_rdma_client_adhoc_user_mr_total` / `dfkv_rdma_client_transient_user_mr_active` | counter / gauge | pool 外实际注册累计 / 当前仍存活的一次性 MR；公开调用返回后 active 必须回到调用前基线 |
| `dfkv_rdma_client_pool_mr_registrations_total` / `dfkv_rdma_client_pool_mr_registration_failures_total` | counter | shared-PD 上真实 `ibv_reg_mr` 次数 / 单 rail 显式 pool 注册失败；包含最终回滚的尝试 |
| `dfkv_rdma_client_pool_mr_active_registrations` | gauge | 进程内仍有 endpoint 引用的 shared-PD MR generations；扩容成功后 anchor/空闲 endpoint 立即释放旧代，在飞旧 endpoint 到下次 acquire/close 才释放，确保旧 range 不中断 |
| `dfkv_rdma_client_max_block_seen_bytes` / `dfkv_rdma_client_declared_max_block_bytes` / `dfkv_rdma_client_connection_min_block_bytes` | gauge | 实际请求高水位 / 逻辑安全上限 / adaptive QP 最小 class |
| `dfkv_rdma_client_connection_class_opened_total{block_bytes,depth}` | counter | adaptive二维class累计新建data QP |
| `dfkv_rdma_client_connection_class_active{block_bytes,depth}` / `connection_class_idle{block_bytes,depth}` | gauge | 当前执行中 / idle pool内的二维class分布；label集合受power-of-two class限制 |
| `dfkv_rdma_client_oversize_rejects_total` | counter | 分配、注册或发帖前因超过声明上限而拒绝的操作 |
| `dfkv_rdma_client_v2_probe_attempts_total` / `dfkv_rdma_client_v2_probe_failures_total` | counter | 必选 v2 bootstrap probe 尝试 / 失败 |
| `dfkv_rdma_client_stale_pool_retries_total` | counter | pooled QP 失败后改用 fresh connection 的重试；含已 post 的 PUT 与直接写入调用方内存的整对象 GET，二者只在连接是 stale pooled（见下一行）时重试 |
| `dfkv_rdma_client_pipelined_read_conns_total` | counter | 以流水线 RDMA READ（每个 QP 最多 16 个 READ 同时在途）建立的连接：client 开了 `DFKV_RDMA_PIPELINED_READS` 且 server 在探测回答里声明了 READ 深度 |
| `dfkv_rdma_client_session_closed_reaps_total` | counter | 复用前或 keepalive 前发现 server 已关闭会话通道而直接丢弃的池化连接（`DFKV_RDMA_SESSION_FD`）；这些连接不再计入 `stale_pool_connections`，也不用等 RC 重试耗尽 |
| `dfkv_rdma_client_stale_pool_connections_total` | counter | 从池里取出复用时即失败的连接数（对端回收了空闲连接或已重启）。这类失败只丢弃该连接，不冷却对端 rail；对端有身份（经 MDS 发现）时同 (peer, rail) 的其余空闲连接一并丢弃，静态地址列表里的对端则不丢弃其余连接，它们各自被发现失效时各计一次；随后的 fresh connection 若也失败才计入 `dfkv_rdma_client_remote_rail_failures_total` 并冷却。持续增长 = server 的空闲回收（`DFKV_RDMA_IDLE_MS`）短于 client 的连接空闲时间，可开 `DFKV_RDMA_KEEPALIVE_MS` |
| `dfkv_rdma_client_peer_unreachable_total` | counter | 新建连接时到对端 bootstrap 地址的 TCP connect 失败的次数（含能力探测）。这是节点级故障：不计任何 rail、不重试（各 rail 拨的是同一个地址），操作返回 `kIOError`，由上层按节点冷却。失败后该地址在 `DFKV_RDMA_REMOTE_RAIL_COOLDOWN_MS`（默认 2 s）内不再拨号 |
| `dfkv_rdma_client_peer_unreachable_skips_total` | counter | 因同一地址刚 connect 失败（上一行的保持期内）而没有发起的建连次数；这些操作和后台节点探测立即返回 `kIOError`，不再各等一次 connect 超时。已建立的连接不受影响 |
| `dfkv_rdma_client_peer_busy_total` | counter | 操作的首次建连被对端在回任何字节之前关闭的次数。server 的 pending handshake 达上限（`DFKV_RDMA_MAX_PENDING_HANDSHAKES`）或不提供所请求的设备时都是这样；不计 rail，等待 20–40 ms 后重试一次：有其他兼容 rail 就换 rail，否则在同一 rail。重试再被关闭才按 endpoint failure 计入 `dfkv_rdma_client_remote_rail_failures_total` 并冷却。持续增长时对照 server 的 `dfkv_rdma_rejected_connections_total` 和 server 日志里的设备拒绝；确认是设备拒绝的 rail 计入 `dfkv_rdma_client_rail_refusals_total` 并冷却 |
| `dfkv_rdma_client_rail_refusals_total` | counter | 对端在某条 rail 上回复前关掉 bootstrap、同一操作随即在另一条 rail 上连上，client 再在原 rail 上拨一次确认仍被关掉：认定它不服务这个设备，把 (peer, rail) 记为 endpoint 失败并冷却。对端只剩这一条可用 rail 时不冷却。确认拨号不计入 `dfkv_rdma_client_peer_busy_total`，连接和每次读最多等 300 ms，超时不算确认。对端恰在确认拨号时也处于握手满载或 receive-pool 压力下时，这里也可能计入；代价有界：基础冷却 `DFKV_RDMA_REMOTE_RAIL_COOLDOWN_MS`（默认 2 s）、之后有恢复探测、最后一条可用 rail 不冷却 |
| `dfkv_rdma_rail_port_active{dev}` | gauge | server 端口检查（`DFKV_RDMA_PORT_CHECK_MS`）最近一次看到的各 rail 端口状态：1 ACTIVE，0 不是，-1 未知或未检查 |
| `dfkv_rdma_inactive_rail_refusals_total` / `dfkv_rdma_rail_down_connections_ended_total` / `dfkv_rdma_rail_port_changes_total` | counter | server：因 rail 端口不是 ACTIVE 而在握手时关掉的连接 / 因端口离开 ACTIVE 而结束的这条 rail 上的连接 / 看到的端口状态变化次数 |
| `dfkv_rdma_client_op_deadline_expired_total` | counter | 操作的期限（`DFKV_RDMA_OP_TIMEOUT_MS`，批量操作为每次尝试的批量期限）在建连途中用完的次数：操作以 `kIOError` 超时，不记对端不可达、不冷却任何 rail |
| `dfkv_rdma_client_rails_port_down` | gauge | 当前因端口不是 ACTIVE 而退出选择的本端 rail 数（`DFKV_RDMA_PORT_CHECK_MS`） |
| `dfkv_rdma_client_port_down_events_total` | counter | 本端 rail 因端口离开 ACTIVE 而退出选择的次数 |
| `dfkv_rdma_client_async_events_total{dev,event}` | counter | 读到的网卡异步事件（`IBV_EVENT_*` 的名字），开关关闭时也计数 |
| `dfkv_rdma_client_device_suspect_total{dev,reason}` | counter | rail 的设备可能已复位的信号：`fatal`（DEVICE_FATAL），`open`（新建 endpoint 失败于 `ENODEV`/`EIO`） |
| `dfkv_rdma_client_device_dead{dev}` | gauge | 因设备复位退出选择、等待重建的 rail（`DFKV_RDMA_DEVICE_REBUILD`） |
| `dfkv_rdma_client_device_rebuilds_total{dev}` | counter | 在复位后的设备上重建完成的次数 |
| `dfkv_rdma_client_dead_release_failures_total` | counter | 在已证明失效的设备上释放失败、被丢弃而没有中止进程的资源（进程级） |
| `dfkv_rdma_client_local_resource_failures_total` | counter | 因本进程资源问题失败的操作：调用方 buffer 注册失败且 errno 为 ENOMEM/EAGAIN/EFAULT/EINVAL（locked-memory 上限、地址不可 pin），或建 endpoint 时 fd/内存耗尽（EMFILE/ENFILE/ENOMEM）。不计 rail、不冷却对端；建连时 `socket()` 或建 endpoint 时 fd 耗尽会先用对端在其他 rail 上的空闲连接，ENOMEM（可能只是该设备的 QP/CQ 上限）会在其他 rail 上新建连接（见下一行），都不行时操作返回 `kResourceExhausted`。**按失败的那次打开计**，不按操作：一次操作在几条 rail 上各失败一次就计几次；日志里有 `cannot register a caller buffer` / `cannot create an endpoint` 及 errno。其他 errno 仍按本地 rail 故障处理。>0 先查本进程的 `ulimit -l` / `LimitMEMLOCK`，再查 `LimitNOFILE` |
| `dfkv_rdma_client_local_resource_rail_fallbacks_total` | counter | 建 endpoint 因本地资源失败后，改在其他 rail 上完成的操作（fd 耗尽时用已有的空闲连接，ENOMEM 时新建） |
| `dfkv_rdma_client_local_resource_held_total` | counter | 直接以 `kResourceExhausted` 失败的操作：选中的 rail 没有空闲连接，且它和其他可用 rail 最近 2 秒内建 endpoint 都 ENOMEM 过（进程的 locked memory 用完），不再拨号，免得每次操作都让对端看到 N 个半途而废的握手；处于这 2 秒内的 rail 上已有的空闲连接照常使用 |
| `dfkv_rdma_client_cross_rail_retries_total` | counter | 无 label；client-local `kRailFailure` 后真正启动的 cross-rail logical retry。GET 可重试未完成 work；PUT 只有 request 未 post 才计入 |
| `dfkv_rdma_client_cross_rail_retry_successes_total` | counter | 无 label；上述 retry 最终成功的 logical operation 数 |
| `dfkv_rdma_client_cross_rail_retry_exhausted_total` | counter | 无 label；上述 retry 的第二次物理尝试仍失败 |
| `dfkv_rdma_client_completion_timeouts_total` | counter | 消耗完一次绝对 completion-window deadline 的窗口；部分完成不重置预算 |
| `dfkv_rdma_client_endpoint_budget{kind}` / `qp_budget{kind}` / `wr_slot_budget{kind}` / `registered_slot_bytes_budget{kind}` | gauge | `kind=used|limit` 的进程级资源占用/上限；registered bytes 按每 endpoint 的 server receive + pull 两个 arena 计量 |
| `dfkv_rdma_client_resource_budget_raises_total` | counter | 预算自适应放大次数（随 ring 采纳，只增不减；任一预算 env 显式设置时恒为 0） |
| `dfkv_rdma_client_admission_failures_total` | counter | 本进程预算饥饿导致的操作失败（返回 kResourceExhausted，对端从未被拨号、不进冷却）；应恒为 0，>0 说明预算仍小于扇出需求 |
| `dfkv_rdma_client_depth_refunds_total` | counter | 服务端 clamp depth 后按协商值退还 WR/注册字节预算的连接数（首连按上限探测，后续按学习值建连） |
| `dfkv_rdma_client_topology_nodes` | gauge | 最近一次喂给预算自适应的 ring 规模（0=从未提示或自适应关闭） |
| `dfkv_rdma_client_peer_topology_updates_total` | counter | 无 label；接纳的新 peer immutable topology generation 数；同 generation 重复通知不增加。generation 按成员各自的拓扑条目计算，某个成员变化时只有它自己计数，其余成员的重复发布不计 |
| `dfkv_rdma_client_no_compatible_rail_total` | counter | 无 label；configured tiers 因 absent/incomplete peer topology 或最高可用交集为空而返回 `kNoCompatibleRail` 的次数；不计 peer health failure |
| `dfkv_rdma_client_stale_generation_reaps_total` | counter | 无 label；因 Conn 的 peer generation 落后而拒绝 take/repool/keepalive、并经 single-owner lifecycle 回收的 endpoint 数；只涉及拓扑条目发生变化（或被移除、重新加入）的那个成员的连接 |
| `dfkv_rdma_client_keepalive_attempts_total` / `dfkv_rdma_client_keepalive_successes_total` / `dfkv_rdma_client_keepalive_failures_total` | counter | idle pooled QP 的保活尝试 / 成功 / 失败并退役连接；仅 `DFKV_RDMA_KEEPALIVE_MS>0` 时增长。保活失败只丢弃该连接，不冷却对端 rail（不计入 `remote_rail_failures_total`）；一轮保活逐条探测，每条只在自己被探测时离开连接池 |
| `dfkv_rdma_client_rail_conns_total{dev}` / `dfkv_rdma_client_rail_selections_total{dev}` | counter | 每 rail 新连接 / 准入分布 |
| `dfkv_rdma_client_rail_put_ops_total{dev}` / `dfkv_rdma_client_rail_put_bytes_total{dev}` | counter | 每条本地rail成功完成的逻辑PUT对象数/有效payload字节；跨多个WR的SG对象只计一个逻辑op |
| `dfkv_rdma_client_rail_get_ops_total{dev}` / `dfkv_rdma_client_rail_get_bytes_total{dev}` | counter | 每条本地rail成功完成的逻辑GET对象数/实际返回payload字节；用于验证rank-local affinity是否真正承载业务流量 |
| `dfkv_rdma_client_rail_inflight{dev}` / `dfkv_rdma_client_rail_credits_available{dev}` | gauge | 当前已租 / 可用 request credits |
| `dfkv_rdma_client_rail_credits_exhausted_total{dev}` | counter | 因 local candidate credit 不足跳过次数 |
| `dfkv_rdma_client_rail_latency_ewma_us{dev}` | gauge | 选轨用的延迟估计：每 64 KiB 的操作延迟（不含建连）的 EWMA，每 1 s 无样本减半 |
| `dfkv_rdma_client_rail_background_probes_total` | counter | 冷却期满的 rail 由后台线程发出的 recovery probe（对已连接节点的一次 keepalive）；成功计入 `dfkv_rdma_client_rail_recoveries_total{dev}` |
| `dfkv_rdma_client_rail_errors_total{dev}` / `dfkv_rdma_client_rail_consecutive_errors{dev}` | counter / gauge | **仅本地** device open、QP transition、MR register/refresh、post/CQ verbs API 失败和 local WC 累计 / 当前连续值 |
| `dfkv_rdma_client_endpoint_errors_total{dev}` | counter | peer bootstrap/不可达/probe/protocol/receive-segment/decode、remote/RNR/retry/timeout/flush WC；credit 已归还，**不惩罚或恢复 rail**。没有 local-error WC 的 completion deadline 也属 endpoint，不触发 cross-rail retry |
| `dfkv_rdma_client_rail_quarantines_total{dev}` / `dfkv_rdma_client_rail_quarantined{dev}` / `dfkv_rdma_client_rail_recovery_probe{dev}` / `dfkv_rdma_client_rail_recoveries_total{dev}` | counter / gauge / gauge / counter | 连续 3 次 client-local failure 后隔离（默认 cooldown 5000 ms）/ 隔离状态 / cooldown 到期后唯一真实 probe / probe 成功恢复；probe local failure 重启 cooldown，endpoint failure 只释放 probe ownership |
| `dfkv_rdma_client_numa_fallbacks_total{reason="caller_unknown\|no_local_rail"}` | counter | `DFKV_RDMA_NUMA=1` 无法建立 local mask 而回退全部 enabled rails |
| `dfkv_rdma_cq_completions_total` / `dfkv_rdma_cq_errors_total` | counter | client CQ 完成 / 错误完成 |
| `dfkv_rdma_client_rail_wc_errors_total{dev,status}` | counter | 因 work completion 失败而销毁的连接数，按本地 rail 和首个失败完成的 status 分解（随后被 flush 的完成不重复计，所以不等于 `dfkv_rdma_cq_errors_total`）。series 在首次出现时才导出，告警用 `dfkv_rdma_cq_errors_total`；对端地址不做 label，见 §3.7 |
| `dfkv_rdma_client_pipeline_depth` | gauge | 握手解析后的有效 pipeline depth |
| `dfkv_rdma_client_pool_connections{lane,state=\"idle|active\",dev}` | gauge | endpoint 按最后/当前 operation lane、ownership state 和 rail 分解；active 不含已隔离连接 |
| `dfkv_rdma_client_peer_connections{peer,dev}` | gauge | 当前 live endpoint 按 MDS stable peer identity 与本地 rail 分解；连接销毁后 series 删除，cardinality 受 endpoint budget 上限约束 |
| `dfkv_rdma_client_pool_limit` | gauge | 每 peer/pool 的 idle retention 基线（默认 8）；未显式配置 POOL_MAX 时 data pool 可短期保留至 64，超过基线的应用空闲连接在 60 秒后退休，仍受进程资源预算限制 |

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

SGLang HiCache插件另导出：

| 指标 | 类型 | 含义 |
|---|---|---|
| `dfkv_hicache_parallel_identity{tp_rank,physical_rank,pcp_rank,dcp_rank,storage_pcp_rank,primary_dev}` | gauge | 每个connector进程的并行坐标、node-local物理rank、存储PCP writer坐标和primary rail；`physical_rank=-1`表示affinity关闭 |
| `dfkv_hicache_exist_probes_total{tp_rank,result}` | counter | `result=full_miss|partial_prefix|full_hit`的exist probe次数 |
| `dfkv_hicache_retain_keys_total{tp_rank,result}` | counter | retention SET 的 key 数，按结果：`protected`、`miss`（尚未写入，TTL 挂起等写入时施加）、`unprotected`（在服务端策略内未保护）、`denied`、`unsupported`、`error`。后三者表示请求的保留时长没有兑现，同时打限频 WARNING |
| `dfkv_hicache_exist_probe_pages_total{tp_rank}` | counter | 候选page总数 |
| `dfkv_hicache_exist_present_pages_total{tp_rank}` | counter | probe批内任意位置存在的page总数 |
| `dfkv_hicache_exist_contiguous_pages_total{tp_rank}` | counter | 从首页开始连续、可供SGLang使用的page总数 |
| `dfkv_hicache_parallel_context_fallbacks_total{tp_rank}` | counter | 创建存储后端时读不到 SGLang parallel context、存储坐标改取自 `HiCacheStorageConfig` 或默认值的次数（每个存储后端记 1）。当前 SGLang 上应为 0；非 0 的 rank 按 `tp_rank` 选举 MLA 写入者，启动日志里有各坐标的来源 |
| `dfkv_hicache_host_writes_while_device_direct_pages_total{tp_rank}` | counter | 主 KV 以 device-direct 形态存储时，`batch_set_v1` 以 host 形态写入的页数（controller 在注册 device pool 之后退回 host 路径，例如混合模型 sidecar 注册失败）。这些页存在裸 key 下，device 形态的存在性探测查不到；写入照常进行，首次打一条 WARNING。正常应为 0 |
| `dfkv_hicache_length_mismatch_keys_total{tp_rank}` | counter | 读到存储长度与本进程不同的 key 数。每个 key 每个进程只改写（REMOVE 后 PUT）一次，之后只计数；持续上涨说明两种布局的实例共享一个 namespace，需要不同的 `model_revision` 或 `bind_runtime_layout`（见 CONNECTORS.md §2.2） |
| `dfkv_hicache_exist_seconds{tp_rank}` | histogram | connector整次`batch_exists`耗时 |

`present_pages - contiguous_pages`直接表达“后页存在但前缀有洞”；不要用
`prefetched_tokens_total`反推物理RDMA字节。轨道真实读写用上述per-rail bytes。

**retry 语义与告警**：

- `cross_rail` counter 只统计 client-local `kRailFailure` 后真正启动的跨轨 retry；
  pooled endpoint staleness仍由 `stale_pool_retries_total` 单独计数。每个 logical
  operation 捕获一个 peer generation，物理尝试最多 2 次。
- GET/batch/SG 保留已完成 item/window 的 byte-exact staged 结果，fresh 第二次只
  提交未完成集合。PUT 仅在明确 request 尚未 post 时允许换轨；post 后 completion
  不明的 PUT 不 cross-rail replay，避免扩大 ambiguous write 语义。例外是 stale
  pooled QP（出池后无任何 completion 成功即失败，见
  `stale_pool_connections_total`）：请求几乎可以肯定未送达，PUT 与整对象直接 GET
  都在 fresh connection 上再做一次（同一 key、同一字节），计入
  `stale_pool_retries_total`。
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
| `dfkv_connector_gpu_dedup_load_fences_total` / `dfkv_connector_gpu_dedup_load_fence_failures_total` / `dfkv_connector_gpu_dedup_invalidate_stuck_total` | counter | — | 原生 `dfkv_client_gpu_dedup_*` 同名计数的转发（含义见 §3 客户端表）。原生库不输出的计数不上报：`invalidate_stuck` 要等原生库开始输出它之后才有数据 |

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

LMCache 的健康 ping 每次 GET 8 个哨兵 key（`dfkv-ping/0..7`，各 1 字节、只读），
所以每次 ping 在客户端和节点的 GET 计数（含 miss）与访问日志里各多 8 次未命中；
看 GET 命中率时按 LMCache 的 ping 频率扣除。

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
| `dfkv_membership_ring_epoch{group}` | gauge | 按 placement 内容计算的 FNV-1a ring epoch，取自 `dfkvctl ring --json` 的 `placement_epoch`（主要用于变更关联；Prometheus float 可能舍入 64-bit 低位） |
| `dfkv_membership_registration_revision{group,node}` | gauge | etcd 中该节点注册 key 的最新 `mod_revision`，可关联停止更新/大面积重注册事件 |

建议 cron/systemd timer 每分钟运行一次，命令本身用 `--timeout 5`；同时对
`dfkv_membership_audit_ok == 0` 或脚本退出非零告警。textfile 有上一次成功文件不代表
本次 timer 成功，必须另外监控 timer exit/文件 mtime；不要只看陈旧的 `ok=1`。
split-brain、MDS/etcd 漂移和自报不一致均由脚本以 `critical` + exit `2` 给出，
JSON `issues[].code/message` 包含具体 endpoint/node 和处置方向。

### 3.7 RDMA 失败完成的归因（日志行 + 分 status 计数）

一次 work completion 失败（例如 `IBV_WC_RETRY_EXC_ERR`，status 12）后，要回答的是
“哪个对端、本地哪条 rail、哪个操作、什么 status”。计数器回答 rail 和 status，
日志行补上对端和操作：

- 计数：client `dfkv_rdma_client_rail_wc_errors_total{dev,status}`，server
  `dfkv_rdma_rail_wc_errors_total{dev,status}`。`status` 取 verbs 枚举名去掉
  `IBV_WC_` 前缀的小写形式，共 24 个已知值，表外的值一律记为 `other`，label 集合
  有上界。常见值：`wr_flush_err`(5)、`rem_access_err`(10)、`retry_exc_err`(12)、
  `rnr_retry_exc_err`(13)、`resp_timeout_err`(20)、`general_err`(21)。
- 这两个 family 的 series 在某个 (`dev`, `status`) 第一次失败时才出现，初值就是 1，
  `increase()` / `rate()` 会漏掉每个组合的第一次失败。**告警用一直存在的总量**：
  server `dfkv_rdma_rail_completion_errors_total{dev}`（或
  `dfkv_rdma_completion_errors_total`），client `dfkv_rdma_cq_errors_total`；分 status
  的 series 用来在告警之后做分解。
- 对端地址数量没有上界（server 侧是来来去去的 client），所以**不做 label**，只进
  日志。

client（WARN，连接因失败完成被销毁时打印）：

```
rdma client CQ failure peer=<ip:port> peer_id=<MDS identity 或 -> dev=<本地 rail> op=<操作>
  status=<数字> status_name=<名字> vendor=<vendor_err> wr_id=<slot> qp=<本端 QPN>
  lane=<payload|sg|control> pooled=<0|1> idle_ms=<n> verdict=<rail|endpoint|neutral> suppressed=<n>
  peer_rail_charged=<0|1>
```

- `op`：失败时该 slot 上在途的请求，取值 `put` / `lease_put` / `pull_range` /
  `pull_release` / `exist` / `lookup` / `remove` / `retain` / `members` /
  `keepalive` / `unknown`。GET 的 one-sided READ 阶段记为 `pull_range`。
- `pooled=1` 表示这条连接此前完成过业务操作（来自连接池）；`idle_ms` 是距上一次
  业务操作完成的时间，`pooled=0` 时为 0。
- `verdict` 是这次失败记到了哪一边：`rail` 对应
  `dfkv_rdma_client_rail_errors_total`，`endpoint` 对应
  `dfkv_rdma_client_endpoint_errors_total`。
- `peer_rail_charged=1` 表示这次失败计入对端 rail 的健康
  （`dfkv_rdma_client_remote_rail_failures_total`，可能触发冷却）；`0` 表示不计：对端没有身份、连接从池里取出后无任何 completion 成功（stale pooled），
  或是保活探测失败。`verdict=endpoint` 只说明不算本地 rail 的错误。
- 只有 deadline 超时、没有失败完成的窗口不打印这一行，看
  `dfkv_rdma_client_completion_timeouts_total`。保活探测最多等 100 ms，而对端 QP
  已不存在时重传耗尽要更久，所以这种探测多数以超时结束，只计入
  `dfkv_rdma_client_keepalive_failures_total`；`op=keepalive` 只在失败完成赶在这
  100 ms 内到达时出现。
- 这一行限频（每个 transport 实例各自计）：10 秒窗口内同一
  (对端, rail, status, vendor) 只打一行，同一 status 下 vendor 不同算不同的失败、各打
  一行；一个窗口最多 32 个不同组合，超出的组合在该窗口内不打印。没打出来的次数
  不分对端和 status 合计成一个数，由下一条打印出来的行在 `suppressed` 里带出（可能
  隔很久、也可能是别的对端的行）。计数器不受限频影响，数量以计数器为准。

server（WARN，每条失败的连接各打一行，不限频；原有字段和顺序不变，在行尾追加
`status_name` 起的 4 个字段）：

```
rdma server CQ failure loop=<sync|uring> status=<数字> opcode_raw=<n> vendor=<vendor_err> wr_id=<slot>
  qp=<本端 QPN> peer_qp=<对端 QPN> depth=<n> free_send=<n> queued_recv=<n> posted_sends=<n>
  status_name=<名字> dev=<本地 rail> peer=<ip:port> age_ms=<连接存活时间>
```

建连行 `rdma conn: ... qp=<n> peer_qp=<n>` 同样追加了 `dev=<本地 rail> peer=<ip:port>`。
失败完成不保证带有效 opcode，所以 server 行没有操作名，只有 `loop` 和原始字段。

读法：

- client 行的 `qp` 就是 server 两条日志里的 `peer_qp`，可以把同一条连接在两端的记录
  对上。client 行的 `peer` 是 server 的 bootstrap 地址；server 行的 `peer` 是 client
  的 IP 加这条连接 bootstrap 时用的临时端口。
- `retry_exc_err` 且 `pooled=1`、`idle_ms` 不小于 server 的空闲回收窗口
  （`DFKV_RDMA_IDLE_MS`）：通常是 server 已回收这条空闲连接，对照该 server 的
  `dfkv_rdma_idle_reclaims_total`、`dfkv_rdma_segment_evictions_total`。
- `retry_exc_err` 且 `pooled=0`：新建连接上的请求始终没有送达，优先查这条 rail 到
  该对端的路径，以及对端同名 rail 的状态。
- 同一 `dev` 上对多个对端同时出现，指向本地 rail 或其上游；只有一个对端，指向那个
  对端。

## 4. 性能保证

- 热路径计数：`std::atomic` relaxed `fetch_add`，零锁零分配（与既有模式一致）。
- 延迟：1/64 采样（`Sampler` 掩码），仅采样到的 op 读 vDSO 时钟。
- HTTP `/metrics`：独立端口 + 线程，仅 scrape 时把原子读成文本。
- 客户端计数搭在 `PeerHealth` 既有锁段上（无新增锁）；per-peer 仅错误路径（罕见）。
- RDMA 失败完成的归因（§3.7）只在失败之后执行：成功路径不加锁、不分配、不格式化；
  日志格式化和 client 的限频锁都在连接已经要销毁的分支里，新增的计数器和状态放在
  各结构原有成员之后，不挪动请求路径上的热计数器。
- 插件轮询线程睡眠态、按间隔触发，不在 per-batch/per-page 路径。
- 验证：每 PR 跑 RDMA Soft-RoCE loopback + ThreadSanitizer 全绿。
