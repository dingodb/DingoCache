# dfkv 独立上线 runbook（与现网 dingo-cache 混部）

> 在 GPU 节点上**独立**起一套 dfkv KV 缓存集群供 SGLang HiCache 用，与现网
> `dingo-cache` 混部互不干扰。**对现网 dingofs/sglang 命名空间、dingo-cache 进程、
> MDS、对象存储一律不动（只读边界）**；dfkv 用独立端口/目录/进程。
> 前提：GLM-5.1 = MLA（每页 KV ≈ 2.74 MiB 单对象、跨 TP 复制、仅 tp_rank0 写）。
> 已在 400G InfiniBand 上端到端验证（两端零拷贝；单口 GET ~93% 线速为
v1.35 two-sided 数据面实测，v2 one-sided 需重测）。

> **本文只讲 dfkv 集群自身的部署;各推理引擎如何对接/配置 dfkv(HiCache/vLLM/LMCache + 客户端配置总表)见 [docs/CONNECTORS.md](CONNECTORS.md)。**

---

## 0. 网络模型（关键）

dfkv 把**控制面**与**数据面**解耦：

- **控制面 = TCP + 两边 SEND/RECV**：bootstrap TCP 只交换设备/QP/receive-segment 描述；RDMA QP 的 request descriptor 有界，response buffer 显式预留 `18-byte prefix + 32 KiB`，使 32-KiB `Members` 在 control lane 完整返回；更大响应直接失败、不截断。
- **payload = one-sided RDMA**：v2 PUT 用 `RDMA_WRITE_WITH_IMM` 直落 server 共享 receive-segment slot，GET 由 server `RDMA_WRITE` 到 client 提交的 `{addr,rkey,len}`。数据 fabric 无需 IP。
- **设备发现**：留空时两端各选本地首个 port 1 `ACTIVE` HCA；显式逗号白名单才开启多轨轮转，多 fabric 节点须过滤到两端同名且互通的 fabric。
- **失败策略**：未设 `DFKV_RDMA` 时选择 TCP；一旦选择 RDMA，client/server 必须完成 v2 协商和共享 segment 注册，失败即拒绝启动或连接。

发现：默认走 **MDS 动态发现**（etcd + dfkv_mds，见 §2b）；静态成员表仍作为遗留/单节点备用路径（见 §4-legacy）。无副本（一致性哈希单属主，节点挂 = 该分片 miss → 重算）。

---

## 1. 构建可移植 release 产物（同架构构建机一次）

> v2.11.0 的 canonical Linux 交付是一个自包含 tarball，含 versioned
> `libdfkv.so`/SONAME symlink、全部 CLI/daemon、Python shim、connector 源码、
> deploy 工具和文档。不要从 build tree 手工挑文件。

```bash
git clone https://github.com/dingodb/DingoCache.git && cd DingoCache
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DDFKV_WITH_RDMA=ON -DDFKV_WITH_URING=ON \
      -DDFKV_STATIC_LIBSTDCXX=ON
cmake --build build -j
deploy/package_release.sh build release
tar tzf "release/dfkv-$(cat VERSION)-linux-x86_64.tar.gz"
ldd build/libdfkv.so | grep ibverbs
```
依赖：`libibverbs-dev`（构建期）+ 运行节点装 `rdma-core`。无 RDMA 也可去掉 `-DDFKV_WITH_RDMA` 构建纯 TCP 版。
> QP 信息走 TCP bootstrap 交换（非 librdmacm），所以只依赖 libibverbs，不需要 librdmacm。
> CacheLib/Navy 不是 v2.0.0 的构建依赖，也没有占位 backend/CMake 开关；
> 评估结论、兼容性差距和重开条件见
> [CACHELIB_EVALUATION.md](CACHELIB_EVALUATION.md)。


> ⚠️ **glibc 下限 = 构建机的 glibc**。在 Ubuntu 24.04 / glibc 2.39
> 直接构建的二进制不能部署到 glibc 2.35。CI 的 portable job 和仓库根
> `Dockerfile` 都固定 Ubuntu 22.04 + RDMA + 静态 libstdc++，并构建同一
> canonical tarball。`DFKV_STATIC_LIBSTDCXX` 只静态链接 libstdc++/libgcc；
> libibverbs 仍动态加载 provider，运行节点必须安装 rdma-core。

## 2. 每节点：分发 + 缓存目录

```bash
version=2.11.0
tar xzf "dfkv-${version}-linux-x86_64.tar.gz"
root="$PWD/dfkv-${version}-linux-x86_64"
install -m755 "$root/bin/dfkv_server" /usr/local/bin/dfkv_server
cp -a "$root"/lib/libdfkv.so* /usr/local/lib/
ldconfig
mkdir -p /mnt/disk1/dfkv /mnt/disk2/dfkv /mnt/disk3/dfkv
```
`cp -a` 必须保留 `libdfkv.so -> libdfkv.so.<major> -> libdfkv.so.<version>`
两级 symlink；只拷贝一个被解引用的文件会破坏 ELF SONAME 查找。
容量隔离：`--cap`（总量，按盘均分）自带 LRU 自限；设保守值，确认
`现网用量 + dfkv cap + 预留 < 物理总量`。

## 2b. MDS 层：etcd + dfkv_mds

> **推荐路径**：用 MDS 动态发现时须先起 etcd 集群和 dfkv_mds 副本。

### 2b-1. etcd 集群

使用现有 etcd 或按官方文档起 1/3 节点集群（TTL 30 s 的 keepalive 流量极低）。
建议独立于 Kubernetes etcd，避免写放大干扰。

### 2b-2. dfkv_mds 副本（每个管理节点各一个）

```bash
install -m755 build/dfkv_mds /usr/local/bin/dfkv_mds
```

`/etc/systemd/system/dfkv_mds.service`：
```ini
[Unit]
Description=dfkv MDS (Membership Directory Service)
After=network-online.target
[Service]
Type=simple
# --listen: TCP 监听端口（与 dfkv_server 端口段错开）
# --etcd:   etcd 地址（默认 127.0.0.1:2379）
# --metrics-port: 可选，开 Prometheus /metrics（缺省=不开端口，见 §7 / docs/METRICS.md）；
#   --metrics-bind <addr> 绑定监听地址（缺省 0.0.0.0）
# 加固 env（按需加 Environment=）：DFKV_MDS_FIRST_REQ_MS（首帧 absolute deadline，
#   默认 30000ms，0=关，硬上限 3600000）、DFKV_MDS_MAX_CONNS（并发连接上限，默认 4096）
ExecStart=/usr/local/bin/dfkv_mds --listen 9400 --etcd 127.0.0.1:2379 --metrics-port 9410
Restart=on-failure
RestartSec=2
[Install]
WantedBy=multi-user.target
```
```bash
systemctl daemon-reload && systemctl enable --now dfkv_mds
journalctl -u dfkv_mds -n 5 --no-pager   # 应见 "dfkv_mds listening on 9400, etcd=..."
```

> **无状态**：dfkv_mds 不持久化私有状态，权威状态全在 etcd；进程内 lease
> shortcut 只保留最近使用项，闲置多个 TTL 后可丢弃。重启/增减副本无需协调；
> 节点和客户端各持一份 MDS 端点列表，自动故障转移，无需负载均衡器。
> etcd member key 为 `/dfkv/v1/groups/<group>/members/<id>`。客户端看到的 epoch
> 是**成员放置内容的 hash**，不是 etcd 全局 revision；无关 group 写入和纯统计
> 心跳不会触发环重建。
> MDS 的默认 etcd request timeout 为 2s；一次 heartbeat 的最慢有效顺序路径是
> failed keepalive → lease grant → Put，共 3×2s。node/client registrar 的默认
> socket I/O timeout 为 7s，组合上覆盖这 6s server budget；不要把 caller timeout
> 调到单个 etcd timeout。
> MDS 的 `/healthz` 是纯进程 liveness（不 probe etcd）：否则一次秒级 etcd
> 抖动会让 kubelet 同时重启全部 MDS 副本，CrashLoop 比抖动更久，重注册风暴
> 还会引发 lease 集体过期。`/readyz` 保留 etcd 依赖检查（活着但探不到 etcd
> 的 MDS 无法服务 membership，必须摘出路由），但经 TTL-debounced probe
> （`DFKV_MDS_PROBE_CACHE_MS`，默认 2500 ms；0=恢复逐请求现场 probe；钳制
> 上限 600000 ms），kubelet 抓取节奏不会放大成 etcd 读负载，not-ready 副本
> 每 TTL 最多重探一次。运行期 etcd 不可达时 `/readyz` 503 使 scheduler
> 摘流，etcd 恢复后同一进程自动回 200；`dfkv_mds` 不因运行期 etcd 短暂故障
> 退出。

> **混合代际滚动（v1.x ↔ v2）**：`DFKV_MDS_ACCEPT_LEGACY=1` 把 MDS 控制面版本闸
> 按已知 epoch 放宽一档——v1.x 节点/客户端以旧 42/10 字节帧直接被服务（操作码、
> MemberInfo 载荷、etcd schema、租约语义跨代一致，status 字节按 v1 枚举顺序回声）。
> 默认关闭 = 历史严格行为（旧 epoch 帧立即拒连）。这**只打通 MDS 控制面**：数据面
> 仍严格 epoch 6/7，v1/v2 节点与客户端必须各属一个 group（环级隔离），客户端版本
> 跟随自己的环。迁移期配好后用 `dfkv_mds_legacy_frames_total` 观察旧流量，归零后
> 撤掉该 env 恢复严格模式。


## 3. 每节点：systemd unit
如启用示例 quota 文件，先创建（已存在时绝不覆盖）：
```bash
install -d -m0755 /etc/dfkv
test -e /etc/dfkv/tenant-quotas ||
  install -m0644 /dev/null /etc/dfkv/tenant-quotas
```


`/etc/systemd/system/dfkv.service`（参考 **xb01-0064 生产配置**（8 条 400G IB 轨/主机）映射到 v2.0.0 大生产参数：**v1.37 生产开了什么现阶段保留什么**；**v2 才落地的能力（SYS_URING/CONVoy 合并/监听加固/slab 水位线）设置为 drop-in 注释默认留**。更小形（单 HCA、无 RAM tier）只需对 `--rdma-dev`、`--ram-tier`/`--ram-tier-bytes` 做减法）：

> **xb01-0064 (v1.37-1.40) → v2.0.0 映射关系**：
> | 项 | xb01 生产 (v1.37/1.40) | 本示例 (v2.0.0) | 变动 |
> |---|---|---|---|
> | `--rdma-dev` | 8 轨全列 | 8 轨全列 | 一致 |
> | RDMA depth | `DFKV_RDMA_DEPTH=32` | `DFKV_RDMA_DEPTH=4` | 新默认 4 对齐 client/server——depth=1 会 clamp 批处理窗口，pipeline GET 退化 3-4×、突发 PUT 批失败致 L3 prefetch miss（热轮吞吐 -29.8%，depth=4 后仅 -2.9%）；depth>1 膨胀每连接 segment lease（depth×slot），过大在高并发时易耗尽共享 segment |
> | RDMA_NUMA | `1` | 保留 `DFKV_RDMA_NUMA=1` | 一致 |
> | RAM tier | on + 1TiB | on + 1TiB（`--ram-tier-bytes 1099511627776`） | 一致 |
> | SERVER_URING | 开，depth=32 | 默认关，经 drop-in 打开（须编 `-DDFKV_WITH_URING`） | v2 保留 |
> | READ_COALESCE | 开 | 默认关，经 drop-in 打开（TCP 端共读 convoy） | v2 新增 |
> | 监听加固 | — | `DFKV_TCP_FIRST_REQ_MS=30000` + `DFKV_METRICS_MAX_CONNS=64` | PR#240 新增 |
> | slab 驱逐水位 | — | `DFKV_SLAB_EVICT_HIGH_PCT=92` / `LOW_PCT=88` | v2 新增（默认 92/88，显式解释为风控用途） |
> | systemd Timeout | `TimeoutStartSec=600` / `TimeoutStopSec=600` | 同上 | **arena MR 注册时间在 v2.0.0 显著拉长**（≥64 GiB arena 默认 90s 不够），必须与 arena 大小同步扩大 |
> | OOM | `OOMScoreAdjust=-500` | 保留 | 一致 |
> | tenant quota | `DFKV_TENANT_QUOTAS_FILE` + default=0 | 保留 | 一致 |
```ini
[Unit]
Description=dfkv KV cache node (SGLang HiCache L3, 8x400G)
# 启动顺序：在 MDS 之前动 After；v2 首注册 deadline 60s，必须先起 MDS 再动 server
After=network-online.target dfkv-mds.service
Wants=network-online.target
[Service]
Type=simple
# v2 shared segment 示例：RDMA depth=4（默认，slot=4MiB=min(client MAX_BLOCK_BYTES 4MiB, server --max-msg 32MiB)）、segment=16 GiB
# （depth=4 时 ~1024 data QP；depth>1 膨胀 lease，高并发易耗尽，CONNECTORS §1.2.1
# 的公式按 peak live + pooled QP 复算）。
Environment=DFKV_RDMA_DEPTH=4
Environment=DFKV_RDMA_RECV_SEGMENT_SIZE=17179869184
# 8×400G 轨全轨白名单 + NUMA：B200 生产 host 一台带 8 个 HCA 轨；
# server 两端白名单到**同名互通**一组轨；client 的 `DFKV_RDMA_NUMA=1` 后每 rank 优先本 NUMA 轨，
# 本地轨全不可准入时降级重试全部 enabled rail（rail_select.h/rdma_transport.cc:399-413）。
# --rdma-dev 直传 RDMA server，胜过同名 env；client 侧 `DFKV_RDMA_DEV` 另行注入（CONNECTORS §1.2）。
Environment=DFKV_RDMA_NUMA=1
# v2.0.0 监听器加固（PR#240）：首帧 30s absolute deadline 断 slow-dribble,
# metrics 端口连接数上限 64（防连接占用打满）,
# 解析结果经 `dfkv_tcp_first_req_ms` gauge 上报（METRICS.md §3.1）。
Environment=DFKV_TCP_FIRST_REQ_MS=30000
Environment=DFKV_METRICS_MAX_CONNS=64
# slab 满环自噬第一道防线：水位线 92% 主动驱逐到 88% 保留 headroom（默认两行已在默认启用，
# 本行显式解释. high=0 关）。对应)--;
#Environment=DFKV_SLAB_EVICT_HIGH_PCT=92
#Environment=DFKV_SLAB_EVICT_LOW_PCT=88
# 0/unset = unlimited. If FILE is configured it must exist and parse strictly.
Environment=DFKV_TENANT_QUOTAS_FILE=/etc/dfkv/tenant-quotas
Environment=DFKV_TENANT_DEFAULT_QUOTA_BYTES=0
# --port = TCP(bootstrap+TCP数据) 端口; --rdma-port = RDMA bootstrap 端口; --rdma-dev = 数据面 8×400G
# --metrics-port = 可选 Prometheus /metrics（缺省=不开端口；--id/--group 成为指标标签）;
#   --metrics-bind <addr> 绑定监听地址（缺省 0.0.0.0）
# 此生产示例故意不设置 --store-engine/DFKV_STORE_ENGINE：resolved 默认即 slab。
ExecStart=/usr/local/bin/dfkv_server \
  --dir /mnt/disk1/dfkv,/mnt/disk2/dfkv,/mnt/disk3/dfkv \
  --port 28000 --rdma-port 28001 \
  --rdma-dev ib7s400p0,ib7s400p1,ib7s400p2,ib7s400p3,ib7s400p4,ib7s400p5,ib7s400p6,ib7s400p7 \
  --rdma-depth 4 --rdma-numa 1 \
  --ram-tier on --ram-tier-bytes 1099511627776 --ram-tier-shards 16 \
  --cap 6597069766656 --mds 10.0.0.1:9400,10.0.0.2:9400 \
  --metrics-port 28010 --group default --id n57 \
  --advertise 192.168.1.57:28001 \
  --mds-registration-timeout-ms 60000
# v2.0.0 生产扩展能力（默认关，会自动在 systemd drop-in 覆盖）：
# - io_uring async GET/direct read（xb01 生产为 on;
#   需要编 `-DDFKV_WITH_URING`）：
#Environment=DFKV_SERVER_URING=1
#Environment=DFKV_SERVER_URING_DEPTH=32
# - TCP 端共读 convoy 合并：server 同 key 复本位于重复读共享一个 disk read
#Environment=DFKV_READ_COALESCE=1
#Environment=DFKV_READ_COALESCE_RECUR_MS=1000
#Environment=DFKV_READ_COALESCE_TIMEOUT_MS=500
Restart=on-failure
RestartSec=2
# 大 arena（≥64 GiB）RAM tier 预触+RDMA MR 注册都起页走：每 16 GiB arena +5-10s 启动时间。
# xb01 生产 TimeoutStartSec/TimeoutStopSec=600，90s 默认会 OOM-timeout 掉 dfkv；
# 配大 arena（≥64 GiB）时必须同步调大。
TimeoutStartSec=600
TimeoutStopSec=600
# OOM 调整：内存压力下 Linux OOM Killer 倾向杀占内存过多的进程，dfkv 的 RAM arena 会被白锁。
# xb01 生产-500，降低 dfkv 被错杀风险（NVMe cache-process 死了 模型自己还能重算，预 pin 大 RAM 的
# dfkv arena 更不该成为被出手的对象）
OOMScoreAdjust=-500
CPUQuota=1600%
MemoryMax=64G
Nice=5
LimitNOFILE=1048576
LimitMEMLOCK=infinity        # RDMA 需要锁页内存
[Install]
WantedBy=multi-user.target
```

> **`--max-msg` 默认 32 MiB（`32ull << 20`），是 client 声明值上限；client 默认
> `DFKV_RDMA_MAX_BLOCK_BYTES` 为 4 MiB，实际 slot 取 `min(client 声明, server --max-msg)`，默认无需显式设置。**
> 它是 RDMA receive-segment 单 slot 上限；client 在 DCP2 协商时声明自己的
> `DFKV_RDMA_MAX_BLOCK_BYTES`（默认 4 MiB）。若 server `--max-msg` < client 声明值，server
> 拒绝连接（日志 `client declared max block ... above this server's cap ...`），所有 PUT 失败
> 但 exist 不受影响（不走 max_block 协商）。**不要降低 `--max-msg` 除非同时降低所有 client 的
> `DFKV_RDMA_MAX_BLOCK_BYTES`。**
>
> Completion timeout：`DFKV_RDMA_OP_TIMEOUT_MS`（默认 5000）约束单操作；
> `DFKV_RDMA_BATCH_OP_TIMEOUT_MS`（unset/0 = 跟随前者）覆盖 **所有** multi-item
> Cache/Range/Exist、zero-copy 与 SG 窗口。一个窗口共用一个绝对 deadline，
> 部分 CQ completion 不会续期。调用方临时 buffer 在方法返回后即可释放；
> pool 外 MR 已在成功 completion 后解除，失败路径先销毁 QP 再解除。


> **存储/加速开关（见 [ARCHITECTURE.md](ARCHITECTURE.md) §5–7）。解析顺序为 flag > 环境变量 > `slab`；运行时真值经 `dfkvctl ring` INFO 列（`engine=`/`wr=`/`ram=`）和 `dfkv_build_info{engine,write_mode}` 审计。**
> - `--store-engine slab|file`：不设置 flag/env 时所有 store/server 路径默认 `slab`。slab = extent 池 + sparse `slots.tbl` + dirty/clean epoch。**早期格式→v3 tenant-scoped slab 必须使用空缓存目录**；容量、格式或几何不符会拒绝启动而不会改写原数据，也绝不会静默改用 file。`file` 的 48 字符 tenant+object 文件名同样不读取旧 cache，仅作显式诊断/回滚。
> - `--slab-write direct|buffered`（默认 `direct`）：slab 数据面使用 O_DIRECT；文件系统不支持时整店回退 bounded buffered，以 `wr=` 上报真值。
> - `--ram-tier on`（默认关）：arena 内对象走 RDMA 零拷贝；配合 `--ram-write-mode writeback|writearound`（默认 `writeback`）选择 PUT 先入 RAM 再落盘，或 PUT 直写盘、后续整值 GET 直接从 NVMe 读入最终 arena slot。两种模式都等真实落盘结果后才返回 PUT `kOk`。超 extent 大对象走同预算的 dedicated allocation + bounded copy。`--ram-tier-bytes <bytes>` 定总预算；显式请求 RAM 时 allocation 或不支持的 NUMA mode 会拒绝启动，不能静默退成 disk-only。
> - `--ram-tier-numa interleave|off`（默认 `interleave`）：这是当前完整 mode 集；不接受数字 node ID。arena 预触并绑策略，须核 `MemoryMax`。
> - `--ram-flush-threads <n>` / `DFKV_RAM_FLUSH_THREADS`：请求值会提高到至少 shard 数；实际值由 `dfkv_ram_flush_threads` 报告（默认请求=4×盘数，上限 16）。
> - 微调项（env only）：`DFKV_SLAB_TABLE_SYNC_MS` 控制 table sync 节奏（默认 100 ms，0=关；dirty epoch 重启仍无条件冷重置）。
> - 微调项（env only）：`DFKV_SLAB_EVICT_HIGH_PCT` / `DFKV_SLAB_EVICT_LOW_PCT`
>   （默认 92 / 88，占容量百分比；high=0 关闭）：水位线主动驱逐在 demand 之前
>   保留 headroom，是满环时 cross-class extent 互抢（自噬）的第一道防线；对应
>   计数器 `dfkv_slab_watermark_evictions_total` / `dfkv_slab_cold_steals_total`。
> - `--slab-reclaim-ms <n>` / `--ram-reclaim-ms <n>`（默认 50 / 10，`0`=关）：后台预回收和类再平衡。allocator 按 useful bytes + decayed read heat 选择 donor，跳过 pinned extent；通常保留默认。
> - `--slab-granularity <bytes>`（默认 1 MiB）：最小 slot 量子。现有 `slots.tbl` 的 format/geometry 不匹配会 fail closed；修改必须换空目录，服务不会原地冷重建或忽略旧数据。
> - `--put-inflight-limit <n>`（默认 0=关）：并发盘写超过 n 的 PUT 以 kCacheFull 快速拒绝（客户端视为普通 put 失败、不进 cooldown）= 用受控 miss 换掉过载排队尾延迟。RDMA 与 TCP 两条数据路径同受此门约束；RAM 热层的异步 flusher 落盘**不受**此门限制（否则背压会放大为 flush 丢弃）。
> - `--tcp-max-conns <n>` / `DFKV_TCP_MAX_CONNS`（默认 512，硬上限 4096）限制
>   TCP handler/FD；`--tcp-io-timeout-s <n>` / `DFKV_TCP_IO_TIMEOUT_S`（默认
>   60s，硬上限 3600s）回收 silent/半帧连接。达到上限的新连接会被立即拒绝，
>   pooled TCP 客户端重连；监控 §7 的 reject counter。
> - `DFKV_TCP_FIRST_REQ_MS`（默认 30000 ms，0=关，硬上限 3600000）：首帧
>   absolute deadline，从 accept 起第一段完整 request frame 必须到齐。逐字节
>   滴喂的慢连接不再永久占用 handler 槽位（SO_RCVTIMEO 只约束单次 syscall）；
>   每次读的 per-syscall 预算取 min(base timeout, deadline 剩余）。resolved
>   值由 `dfkv_tcp_first_req_ms` gauge 上报（见 METRICS.md §3.1）。
> - RAM 热层 arena 预触 + RDMA MR 注册都在启动期走页：**arena 每 16 GiB 约 +5-10s 启动时间**，配大 arena（≥64 GiB）时同步调大 systemd `TimeoutStartSec`（默认 90s）并让就绪探测等待 metrics 端口。
> GPU 节点推荐：保留默认 slab/direct，按节点突发画像启用 RAM tier 并设置总预算。
```bash
systemctl daemon-reload && systemctl enable --now dfkv
journalctl -u dfkv -n 10 --no-pager
# 应见 "PORT 28000" + "RDMA listening (TCP bootstrap) on port 28001, dev=ib7s400p0"
#      + "dfkv_server MDS registration loop started group=default id=n57 advertise=192.168.1.57:28001"
# 首次成功后另见 "dfkv_server registered with MDS group=default id=n57"
```
> `--mds-registration-timeout-ms` / `DFKV_MDS_REGISTRATION_TIMEOUT_MS` 是首次
> MDS 注册硬截止时间（默认 60000 ms，合法范围 1000–600000 ms，flag 优先）。
> 非整数、越界值均以配置错误退出 2；截止前仍未注册则停止所有 listener 并退出
> 1，由 `Restart=on-failure` 重试，节点不会无限保持 active-but-unregistered。
> `GET /healthz` 反映当前 store/RAM terminal health；`GET /readyz` 还要求本地
> startup 和首次 MDS 注册完成。首次成功后 readiness latch 不因瞬时 heartbeat
> 失败清零，已注册节点继续运行并在 MDS 恢复后续租；必须同时监控
> [METRICS.md](METRICS.md) 的 registrar heartbeat 指标。
> server 的 bootstrap 监听 `0.0.0.0`，靠防火墙限制在内网。优雅关闭已修（`systemctl stop` 约 1s 退出）。
> **多轨与 NUMA**：server 启动时 anchor 白名单内全部 active rail；client 以
> `DFKV_RDMA_RAIL_CREDITS` 的 per-rail credit、归一化 inflight、延迟和本地
> rail 错误分数选轨（分数相同时按发现顺序稳定选择）。留空时两端各选本地首个
> ACTIVE HCA，适合 local device 名不同的单 fabric 主机；显式
> `--rdma-dev` / `DFKV_RDMA_DEV` 是权威白名单并把具体设备选择发给 peer，故
> 生产多 fabric 主机必须在两端过滤到**同名且互通**的一组设备。设备名上限
> 18 字节（v2 dev frame 中 name + capability trailer 共 32 字节），超长的
> 白名单/anchor 设备名在启动时 fail-fast 拒绝，不会截断后误配。
> `DFKV_RDMA_NUMA=1` 在每次 Acquire 读取 caller NUMA；白名单内存在 local
> rail 时严格优先 local mask。caller NUMA 未知或没有 local rail 时回退全部
> 白名单；全部 local rail 都不可准入（被隔离或 credit 耗尽）时也会降级为
> 全 enabled rail 重试一次——local 是严格偏好而不是可用性闸门，不阻止
> progress。receive segment 仍是单块 process-wide 分配，不是
> per-NUMA/per-rail 分片。
>
> **失败域隔离**：TCP bootstrap、peer 不可达、epoch/QP frame/receive-segment
> 不兼容属于 endpoint 失败：立即归还 rail credit，由 client `PeerHealth` 对该
> node 做 bounded cooldown，不增加共享 HCA 的 consecutive failure，也不会让
> node A 隔离 node B 的健康 rail。只有本地 device open、verbs/QP transition、
> post 或 CQ 失败才按 `DFKV_RDMA_RAIL_ERROR_THRESHOLD` 隔离 rail
> `DFKV_RDMA_RAIL_COOLDOWN_MS`；cooldown 到期仍以单 probe 成功恢复，endpoint
> 失败既不误恢复也不重新惩罚 rail。
>
> **v2 segment 预算**：`slot=align4K(4096 + max_raw_payload)`；
> `segment >= Σ(live + client-pool-idle data/control QP × depth × slot)`。lease
> 保留到 QP 销毁或 idle reclaim，不能只数在飞请求。4 MiB/depth=4/16 GiB
> 约容纳 1024 条 data QP（未扣 control lease）；depth=8 时约 512 条，depth=16 时约 256 条。上线先看
> `dfkv_rdma_recv_segment_free_bytes`、`dfkv_rdma_v2_ready` 和
> `dfkv_rdma_v2_conns_opened_total`；free 接近 0 会使新连接被拒绝。

### 3a. 每节点 tenant quota

quota 是**每个 cache node 的物理容量边界**，不是同步的集群全局 reservation。
server 启动时一次性读取（运行期不热加载）：

```text
# /etc/dfkv/tenant-quotas
# <16-lowercase-hex-hash> <uint64-bytes>
6164af84acbc9ade 1099511627776
```

`DFKV_TENANT_DEFAULT_QUOTA_BYTES=0` 或 unset 表示未列出的 tenant 无限；
文件内 limit `0` 也表示该 hash 无限。配置了
`DFKV_TENANT_QUOTAS_FILE` 时，文件缺失、hash 非 16 位小写 hex、非 uint64、
重复 hash 或多余字段都会令节点 fail closed。hash 算法由
`dfkv_common.identity.tenant_hash` 与 native client 共用。原子管理工具不会删除
cache data：

```bash
deploy/dfkv_tenant_quota.py --file /etc/dfkv/tenant-quotas \
  set --tenant tenant-a 1099511627776
deploy/dfkv_tenant_quota.py --file /etc/dfkv/tenant-quotas list
deploy/dfkv_tenant_quota.py --file /etc/dfkv/tenant-quotas \
  remove --hash 6164af84acbc9ade
# 文件在 server 生命周期内 immutable；修改后逐节点 restart 生效。
```

按目标 tenant 的去重后逻辑 working set `B`、写复制因子 `R`、有效 cache 节点数
`N`、实测 Ketama 最大偏斜 `S>=1`、同一 tenant 最大并发新写余量 `W` bytes，
起始值：

```
Q_node = ceil(B * R / N * S) + W
```

quota 计 committed payload bytes；slab slot 内碎片另占物理空间，因此还要满足
`sum(active Q_node × slot_size/payload_size) + 运维预留 <= --cap`。各 quota 是
上限而非预留，超额返回 `kQuotaExceeded`，与全盘/写入门的 `kCacheFull` 分开。
REMOVE 和 store eviction 立即释放 usage；重启从 file 名或 v3 slab record 重建。
受限 tenant 为保证 PUT 的同步 admission 语义会绕过异步 RAM write-back。

### 3b. dfkv_server flag / env 参考矩阵（按类别）

以下均为 v2.0.0（cbe53fa）时刻从 `dfkv_server --help`、源码 `Env*`/`args.get` 解析点与常量实测
整理的**权威服务端配置参考**。flag 优先于同名 env（`--rdma-dev` 直传 RDMA server，其余行为
flag 为 env facade）；未列 flag 的全部 env 均从源码排查就不误报。**客户端
侧（DFKV_RDMA_DEV/DFKV_LIB/DFKV_FANOUT_THREADS 等在同一 host 上的 inference rank 进程）仍然
按 CONNECTORS.md §1 的 vLLM/SGLang/LMCache 通用表设置**，本表只列 **dfkv_server / dfkv_mds**
进程读到的组。

#### a. 核心必填/常用（`dfkv_server --help` 全集）

| flag/env 孪生 | 默认 | 说明 |
|---|---|---|
| `--dir <paths>` | —（必填）逗号分隔多盘 | 每节点 NVMe 数据目录，逗号加多个自动 Ketama 分盘；`--cap` 是**总容量**，按盘均分 |
| `--cap <bytes>` | —（必填） | 节点总容量（LRU 超限触发自裁） |
| `--port <p>` | 0(临时） | TCP bootstrap+数据端口；0=随机 |
| `--rdma-port <p>` | —（RDMA build 必填） | RDMA bootstrap 端口（含 RDMA 时开启） |
| `--rdma-dev` | first ACTIVE 本地 | RDMA 白名单（逗号）；与 `DFKV_RDMA_DEV` 不同：这是 server 的 anchor |
| `--mds <ip:port,...>` | —（生产必须） | 注册进的 MDS 列表；配后须配 `--group/--id/--advertise` |
| `--group` | `"default"` | ring 名 |
| `--id <id>` | 主机名 | server 节点 id（与 `--advertise` 联动） |
| `--advertise <ip:port>` | auto（rdma-port） | 其他节点到本节点的地址 |
| `--weight` | `1` | consistent-hash vnode 数权 |
| `--metrics-port` | omit=off | Prometheus /metrics、healthz、readyz |
| `--metrics-bind` | `0.0.0.0` | metrics/health 绑 IP（内网收敛） |
| `--mds-registration-timeout-ms` / `DFKV_MDS_REGISTRATION_TIMEOUT_MS` | `60000` | 首次 MDS 注册截止（1000–600000） |
| `--store-engine` / `DFKV_STORE_ENGINE` | `slab` | `slab` 为默认；`file` 为显式诊断 fallback |
| `--slab-write` / `DFKV_SLAB_WRITE` | `direct` | slab 数据面 O_DIRECT / buffered；文件系统不支持 DIO 时整店回退 buffered 并以 `wr=` 上报 |
| `--ram-tier` / `DFKV_RAM_TIER` | `off` | RAM 热层（写穿+RDMA zero-copy GET） |
| `--ram-tier-bytes` / `DFKV_RAM_TIER_BYTES` | `16 GiB` | arena 预算（pin 一次即注册） |
| `--ram-tier-numa` / `DFKV_RAM_TIER_NUMA` | `interleave` | NUMA 策略：`interleave`/`off`（不接 node id） |
| `--ram-tier-shards` / `DFKV_RAM_TIER_SHARDS` | `8`, 硬上限 64 | arena 锁分片数（大 arena ≥100 GiB 建议 16） |
| `--slab-granularity` / `DFKV_SLAB_GRANULARITY` | `1 MiB` | slab slot 量子；现有目录 geometry 不符启动拒绝 |
| `--put-inflight-limit` / `DFKV_PUT_INFLIGHT_LIMIT` | `0`=关 | 并发盘写上限，超出返回 kCacheFull 快速拒绝 |
| `--tcp-max-conns` / `DFKV_TCP_MAX_CONNS` | `512`, 硬上限 4096 | cache TCP handler 上限；超限 accept 恒拒 |
| `--tcp-io-timeout-s` / `DFKV_TCP_IO_TIMEOUT_S` | `60`, 硬上限 3600 | per-syscall RCVTIMEO（秒） |
| `--rdma-depth` / `DFKV_RDMA_DEPTH` | `4` | server 提交 QP post 深度；与 client 协商取 `min` |
| `--rdma-numa` / `DFKV_RDMA_NUMA` | `0` | NUMA-aware rail choice（off/1） |
| `--rdma-idle-ms` / `DFKV_RDMA_IDLE_MS` | — | idle connection reaper tick |
| `--rdma-op-timeout-ms` / `DFKV_RDMA_OP_TIMEOUT_MS` | `5000` | per-op RDMA deadline |
| `--server-uring` / `DFKV_SERVER_URING` | `0` | io_uring async-GET（编须 `-DDFKV_WITH_URING`） |
| `--server-uring-depth` / `DFKV_SERVER_URING_DEPTH` | — | uring SQ 深度 |
| `--ram-flush-threads` / `DFKV_RAM_FLUSH_THREADS` | 默认 4×盘数， 钳至 shard 数上限 16 | RAM tier flush 线程 |
| `--slab-table-sync-ms` / `DFKV_SLAB_TABLE_SYNC_MS` | `100 ms`, `0`=关 | slab 元数据 sync 节奏 |
| `--slab-reclaim-ms` / `DFKV_SLAB_RECLAIM_MS` | `50 ms`, `0`=关 | slab 后台回收 tick |
| `--ram-reclaim-ms` / `DFKV_RAM_RECLAIM_MS` | `10 ms`, `0`=关 | RAM tier 后台回收 tick |
| `--log` / `DFKV_LOG` | `INFO` | INFO/DEBUG/WARN/ERROR |
| `--max-msg`（或 `DFKV_RDMA_MAX_PAYLOAD_BYTES`） | `32 MiB` | 单笔 payload 硬上限（RDMA 与 TCP 数据路径同受此限） |
| `--version, -V` / `--help` | — | 打印版本/帮助并退出 |

#### b. RDMA 传输面（均在 dfkv_server 进程读取）

| env | 默认 | 说明 |
|---|---|---|
| `DFKV_RDMA_RECV_SEGMENT_SIZE` | 动态 | v2 共享 receive segment 总大小；`slot=align4K(4096+max_raw_payload)`，容量按 expected live QP×depth×slot 预算 |
| `DFKV_RDMA_CONNECT_MS` | — | IB QP 建连超时 |
| `DFKV_RDMA_IO_MS` | — | 控制面帧读写超时 |
| `DFKV_RDMA_BATCH_OP_TIMEOUT_MS` | 0=跟随 RDMA_OP | multi-item Cache/Range/Exist、SG 窗口总期限 |
| `DFKV_RDMA_POOL_MAX` | — | client 侧 per-endpoint 连接池上限 |
| `DFKV_RDMA_RAIL_CREDITS` | 默认 `64`, 硬上限 4096 | client 每 rail QP 信用数（inflight 上限），>server depth 会回退 |
| `DFKV_RDMA_RAIL_BACKPRESSURE_MS` | `10` | Acquire 失败重试 backoff |
| `DFKV_RDMA_RAIL_COOLDOWN_MS` | `5000` | rail quarantine 冷却期；期间只允许 1 个 recovery probe |
| `DFKV_RDMA_RAIL_ERROR_THRESHOLD` | `3` | 连锁本地 verbs/CQ/post 错误才 quarantine rail |
| `DFKV_RDMA_RAIL_LATENCY_WEIGHT` | — | rail selection 的延迟 EWMA 权重 |
| `DFKV_RDMA_RAIL_ERROR_PENALTY_US` | — | rail selection 的错误分数惩罚微秒 |
| `DFKV_RDMA_IDLE_MS` | — | idle QP 重回收周期 |

#### c. TCP 连接池（client 读，但 server 运维也受影响）

| env | 默认 | 说明 |
|---|---|---|
| `DFKV_TCP_POOL_MIN_CONNS` | `1` | per-node pool 下限 |
| `DFKV_TCP_POOL_MAX_CONNS` | `8`, 硬上限 64 | per-node pool 上限 |
| `DFKV_TCP_POOL_IDLE_MS` | — | idle 连接踢出周期 |
| `DFKV_TCP_POOL_ACQUIRE_TIMEOUT_MS` | — | acquire 超时 |
| `DFKV_TCP_POOL_BACKOFF_BASE_MS` / `DFKV_TCP_POOL_BACKOFF_MAX_MS` | — | stale 再拨指数 backoff |

#### d. slab 存储引擎细节调优

| env | 默认 | 说明 |
|---|---|---|
| `DFKV_SLAB_EVICT_HIGH_PCT` / `DFKV_SLAB_EVICT_LOW_PCT` | `92` / `88`, high=0 关 | 水位线主动驱逐阈值；`dfkv_slab_watermark_evictions_total`/`dfkv_slab_cold_steals_total` 对应 |
| `DFKV_SLAB_COLD_STEAL_WINDOW` | — | 小受害者选择窗口（跨 size class 强占） |
| `DFKV_SLAB_URING_WRITE` | — | slab io_uring 异步 PUT（需要 `-DDFKV_WITH_URING`） |
| `DFKV_SLAB_RECLAIM_MS` | `50` | 背景预回收 tick（见 §3） |
| `DFKV_SLAB_GRANULARITY` | `1 MiB` | slot 量子 |

#### e. RAM 热层细节调优

| env | 默认 | 说明 |
|---|---|---|
| `DFKV_RAM_TIER_EXTENT_BYTES` | — | arena extent 大小（通常保留默认；与 `DFKV_RAM_TIER_BYTES` 比例决定并发度） |
| `DFKV_RAM_TIER_LARGE_RESERVE_BYTES` | — | 超大对象（超出 arena extent）的 dedicated allocation 预算 |
| `DFKV_RAM_TIER_SHARDS` | `8`, 硬上限 64 | 锁分片 |
| `DFKV_RAM_TIER_NUMA` | `interleave` | interleave / off |
| `DFKV_RAM_FLUSH_THREADS` | `4×盘数`，钳至 shard 上限 16 | flush 线程 |

#### f. 同主机/会合复本（node dedup，v2.0.0 新生产杠杆）

| env | 默认 | 说明 |
|---|---|---|
| `DFKV_CLIENT_NODE_DEDUP` | `off` | 主开关：跨 rank/process 的同主机同 key 请求会合（仅 client 侧；vLLM replicated-MLA 拓扑自动开） |
| `DFKV_CLIENT_NODE_DEDUP_GPU` | `off` | GPU 目标和 rendezvous（须 CUDA IPC）；off=host 和 rendezvous |
| `DFKV_CLIENT_NODE_DEDUP_LOG` | `off` | 会合决策 log |
| `DFKV_NODE_DEDUP_SLOTS` | `65536`（2 的幂） | dedup 表容量（power-of-2 强制） |
| `DFKV_NODE_DEDUP_GPU_SLOTS` | — | GPU dedup 表容量 |
| `DFKV_NODE_DEDUP_TTL_MS` | `5000` | dedup entry 过期 |
| `DFKV_NODE_DEDUP_WAIT_MS` | — | follower 等待 leader 上限 |
| `DFKV_NODE_DEDUP_TAKEOVER_MS` | — | leader 失联 follower 接管时间 |
| `DFKV_NODE_DEDUP_ARENA_MB` / `DFKV_NODE_DEDUP_GPU_ARENA_MB` | — | dedup 复用 staging arena 大小（重点检查 GPU 复本顿） |

#### g. 一般基础设施

| env | 默认 | 说明 |
|---|---|---|
| `DFKV_LOG` | `INFO` | log level |
| `DFKV_FANOUT_THREADS` | `32` | client 批量 fan-out 线程数（**worker 侧 dfkv client**；大广播/高并发时增） |
| `DFKV_BATCH_CONCURRENCY` | `0`=auto（auto: `min(max(nodes, 8), 32)`） | batch 分段并发度（**worker 侧**）。auto 模式：每节点 1 个 worker，但**下限 8**（单节点环也获得 8 路并行），上限 32。显式设 >0 则固定值。**单节点环必开 auto 或 ≥8**（否则串行，PUT/GET 仅 0.9 GB/s vs 45 GB/s RDMA 带宽） |
| `DFKV_GET_MISS_RETRIES` | `1` | client GET miss 后发重试次数 |
| `DFKV_PROBE_INTERVAL_MS` | `0`=关 | client 对 server 的主动健康 probe 间隔 |
| `DFKV_PEER_COOLDOWN_MS` | `2000` | PeerHealth endpoint 快速回避窗口 |
| `DFKV_PEER_COOLDOWN_MAX_MS` | `30000` | PeerHealth 回避上限 |
| `DFKV_PEER_BAD_GRACE_MS` | `1000` | PeerHealth 转 bad 前的宽限期 |
| `DFKV_DISK_HASH_WEIGHT` | — | 盘内 Ketama 虚拟节点权重（数据中平衡关键调优，改动需重启） |
| `DFKV_READ_COALESCE` | `0` | TCP 端读 convoy 合并主开关（见 README Recommended tuning 表） |
| `DFKV_READ_COALESCE_RECUR_MS` | `1000` | 复制重现窗口（copy fingerprint 64 字节） |
| `DFKV_READ_COALESCE_TIMEOUT_MS` | `500` | follower 等待 leader 上限 |
| `DFKV_READ_MAX_CONNS` | `8` | client 每节点并行 read-shard 连接上限；server 不读取 |
| `DFKV_READ_SHARD_KEYS` | `16` | client 每个 read shard 的目标 key 数；server 不读取 |
| `DFKV_MDS_IO_TIMEOUT_S` | `60` | MDS 控制面帧读写超时（MDS 进程该） |
| `DFKV_MDS_ETCD_PROBE_MS` | `30000` | MDS 与 etcd 存活探测窗（MDS 进程该） |
| `DFKV_TENANT_QUOTAS_COUNT` | — | tenant 配额文件预期条数（容量校验） |

#### 未收录的常见误配 → 详见 CONNECTORS.md §1.7

客户端设置于 dfkv_server 不生效的 env 包括（不是服务端配置，**将此会静默无效**）：
`DFKV_RDMA_DEV`（server 读 `--rdma-dev`）、`DFKV_NODE_DEDUP_*`（全部 client 侧）、
`DFKV_ACCESS_LOG_*`、`DFKV_CLIENT_*`、`DFKV_METRICS_*`、`DFKV_TRACING_*`、
`DFKV_LIB`/`DFKV_BUILD`（客户端 lib 路径）等。

## 4. 集群成员管理

### 4a. MDS 动态发现（推荐）

节点通过 `--mds` + `--group` + `--id` + `--advertise` 向 MDS 自注册。客户端在
调用 `dfkv_open_v2` 前，把 endpoints、group、poll interval 和可选客户端注册身份
一次性写入 `dfkv_client_options_v2`；构造成功后自动轮询 MDS。epoch
（placement 内容 hash）变化时自动重建加权 Ketama 环。增减节点只需启停
`dfkv_server`，无需修改已运行客户端的配置。

两层离线检测：
- **层 2（权威）**：etcd lease 到期（TTL 30s）→ 下次 MDS poll（默认 3s）→
  placement-content epoch 推进 → 环重建。小幅缩容通常约 TTL+一次 poll；
  偏离 `DFKV_MDS_SHRINK_GUARD_PCT`（默认 50，0=同时关闭 shrink/growth
  两臂，空视图臂常在）的批量视图要走 adoption guard：比较基准是**冻结的
  trusted reference**（不随每跳下移，挡住 diffuse mass-expiry 逐跳蚕食），
  shrink 与对称 growth（停机恢复期的批量重注册）两臂可疑视图都必须以
  **同一值连续 3 次 poll** 才采纳；仍在逐跳变化的缩容/恢复斜坡每次 poll
  重置 streak，到 plateau 才一次采纳、中途不重建环。批量缩容被 guard
  挂起时默认最坏约 30s+3×3s，而不是固定“≤30s”。
- **层 1（快速）**：`PeerHealth` 传输 IO 失败即短路该节点为 miss 并进入 cooldown，不触发环重建

### 4b. 静态成员表（遗留/单节点备用）<a name="4-legacy"></a>

适用于无 etcd 的简单环境或单节点调试。成员 = 所有节点的
`name=<bootstrap-ip>:<rdma-port>`（RDMA）或 `:<port>`（TCP）：
```
n57=192.168.1.57:28001,n58=192.168.1.58:28001,...
```
增减节点 = 改成员字符串并重载 SGLang 端。`dfkv_server` 此时不带 `--mds` 启动。
建议 N≥4 降低单点 miss 影响。

### 4c. 节点排空 / 替换（生产自动化）

`deploy/dfkv_node_replace.py` 把无副本缓存环的替换顺序固化为：读取 MDS + 已注册
client 基线 → 新节点入环 → `dfkvctl stat --all` 就绪 → 连续稳定环观测 → 停旧
节点 → 等 lease expiry、客户端 poll 与（批量缩容时）hysteresis 全部收敛 → 再验
新节点和 client。它不会直接删 etcd key；旧节点由正常 service stop 和 lease
expiry 退出权威视图。等待轮询内的瞬态 SSH/命令失败（超时、OSError 等）在
`--timeout` 上限内自动重试。失败回滚由 `started_new`/`old_stopped` 标记定
边界（远端 systemctl 可能在本地 SSH 超时/中断时已经生效，两个标记都在发起
SSH 前置位）：旧节点未停且新节点未标记启动 = safe abort 直接退出；新节点
已启动但旧节点未停 = 先尽力停掉新节点恢复原环再退出；停旧之后任一步失败
或收到中断，脚本会先重启旧 service、等待其重新入环/就绪，再非零退出——
新节点保留在线，避免回滚再制造一次环缩容。
重复执行时若旧节点已退出且新节点健康会直接成功。

```bash
# 先做只读预检并打印会执行的 SSH/systemctl 动作
deploy/dfkv_node_replace.py --mds 10.0.0.1:9400,10.0.0.2:9400 --group glm \
  --old-id n57 --old-host 10.0.0.57 --new-id n70 --new-host 10.0.0.70 \
  --timeout 180 --command-timeout 10 --min-clients 1 --dry-run

# 执行；state 文件是原子更新的事件审计记录
deploy/dfkv_node_replace.py --mds 10.0.0.1:9400,10.0.0.2:9400 --group glm \
  --old-id n57 --old-host 10.0.0.57 --new-id n70 --new-host 10.0.0.70 \
  --timeout 180 --command-timeout 10 --min-clients 1 \
  --state-file /var/lib/dfkv/node-replace.json
```

前提：目标机已安装并配置同名 `dfkv.service`，执行机可 BatchMode SSH；新旧节点 ID
必须不同（同 ID 会争抢 etcd lease，脚本拒绝）。`--timeout` 是**每个状态迁移**的
上限而不是无限等待；`--min-clients 0` 允许还未升级客户端注册能力的旧车队，但脚本
仍会调用 `dfkvctl clients`，并保证执行前已观察到的 client 在切换后没有消失。

### 4d. 成员漂移 / split-brain 巡检

`deploy/dfkv_membership_audit.py` 分别查询每个 MDS（不是把 endpoints 当故障转移
列表只取一个），并从 etcd v3 gateway 只读 `/dfkv/v1/groups/<g>/members/`：
比较 placement view、节点注册值中的自报 INFO、etcd registration `mod_revision`
和按 C++ 同算法计算的 ring content epoch。任一 MDS 不可达、视图分裂、MDS/etcd
成员或自报不一致均退出 `2`；工具/解析错误退出 `1`；健康退出 `0`。

```bash
deploy/dfkv_membership_audit.py \
  --mds 10.0.0.1:9400,10.0.0.2:9400,10.0.0.3:9400 \
  --etcd http://10.0.0.4:2379 --group glm --timeout 5 \
  --output /var/lib/dfkv/membership.json \
  --prom-output /var/lib/node_exporter/textfile/dfkv_membership.prom
```

JSON 中保留每个 MDS 的 ring epoch、完整成员自报和每节点 registration revision，
适合定时任务留档；Prometheus textfile 指标与告警建议见
[METRICS.md](METRICS.md)。etcd gateway 和 MDS 都是无鉴权内网接口，巡检机只开
只读网络访问，不向公网暴露。

### 4e. F10 clean-epoch cutover and rollback boundary

Tenant identity changes every persistent and transport boundary: native TCP
epoch 6, RDMA epoch 7, the 50-byte request prefix, 48-hex file names, and slab
format v3. There is no old wire/disk decoder, dual write, or cache migration.
Old and v2 servers therefore **must never share a client-visible ring**.

Hard invariants:

- Give the candidate distinct MDS endpoints/group, cache/RAM directories,
  service names, data ports, and metrics ports. Never start v2 against an old
  cache directory, and never clear or rewrite that directory in place.
- Switch a drained, compatible client cohort as one unit: connector package,
  `libdfkv.so`, namespace/layout code, and MDS/member endpoints. A mixed client
  artifact can speak the wrong epoch even if its endpoint is correct.
- Keep the old binaries, unit files, endpoint configuration, and cache
  directories unchanged until the rollback window closes. Cache data is
  disposable operationally, but deleting it destroys the only fast rollback.

Cutover procedure:

1. Record the old binary/package hashes, unit files, environment, MDS group,
   endpoints, ring/client snapshots, and cache directories. Freeze changes to
   that cohort for the duration of the cutover.
2. Allocate candidate-only endpoints and **new, initially empty directories**.
   Start candidate etcd/MDS/server services without stopping the old group.
3. Pass `/readyz`, ring/client audit, TCP and RDMA byte-roundtrip, connector
   cold-write/restart-hot-read, and output-correctness gates against only the
   candidate group. A format rejection or unexpected old object is a hard stop.
4. Drain one compatible client cohort, atomically replace its complete client
   artifact/config, point it at the candidate group, then restart it. Do not
   make old and candidate endpoints simultaneous failover peers.
5. Observe readiness, errors, miss/refill convergence, output equality, and
   registered-client membership. Promote further cohorts only after the
   observation window passes.
6. Drain and stop the old cohort only after every routable client has moved.
   Retain its stopped services and directories through the rollback window.

Rollback trigger and procedure:

1. Trigger rollback on protocol/format rejection, byte mismatch, inference
   output mismatch, persistent readiness loss, or a breached load/error gate.
2. Stop new requests to the affected cohort and drain/terminate its candidate
   clients. Do not route an old client to v2 or a v2 client to the old group.
3. Restore the old connector, `libdfkv.so`, namespace code, and old endpoints
   together; restart the old services if they were stopped. Verify their saved
   hashes, ring, registered clients, and an old-format hot read.
4. Stop only the isolated candidate services. Preserve their directories,
   logs, audit artifacts, and etcd prefix for diagnosis; never copy v3
   files/slab metadata into the old directories.
5. After the retention decision, move the retired cohort to an archive path.
   Directory reuse always means a new empty path, never an in-place downgrade.

## 5. 上线顺序 + 冒烟（无需 GPU）

1. （MDS 路径）起 etcd，再起所有 `dfkv_mds` 副本（§2b），确认日志 "listening"。
2. 所有节点起 `dfkv_server`（§3），确认 PORT + "registered with MDS" 日志。
3. 冒烟（任一能访问内网的机器）：
   ```bash
   dfkv_smoke --members n57=192.168.1.57:28000 --size 2752512                          # TCP
   DFKV_RDMA=1 DFKV_RDMA_DEV=ib7s400p0 dfkv_smoke --members n57=192.168.1.57:28001 --size 2752512
   ```
4. 端到端零拷贝校验（插件 → libdfkv → RDMA → server，验证 payload 直落缓冲）：
   ```bash
   DFKV_RDMA=1 DFKV_RDMA_DEV=ib7s400p0 DFKV_MEMBERS='n=192.168.1.57:28001' \
     python3 test/python/rdma_e2e_validate.py    # 期望 RESULT: ZERO-COPY RDMA E2E OK
   # 随后在 server /metrics 确认 v2_ready=1、v2_conns_opened_total 增长且 free_bytes 有余量
   ```
5. 压测（可选）：`DFKV_RDMA=1 DFKV_RDMA_DEV=ib7s400p0 dfkv_bench --members ... --size 2752512 --count 8000 --threads 64`。
6. 在**一个受控 SGLang 副本**上切 `dynamic` 后端，发共享长前缀请求看命中上涨，确认后推广。

### 5b. 候选版本负载回归门
`DFKV_BENCH_STALL_MS=<毫秒>` 只由 `dfkv_bench` 读取；超过阈值的 GET batch
会带 wall-clock 时间戳写 stderr，便于和 server/NVMe/IB 指标对齐。它不是
`dfkv_server` 环境变量。


`deploy/dfkv_load_regression.py` 对 baseline/candidate 使用完全相同的 size/count/
threads/batch/transport 环境，先 warmup，再各跑多轮 `dfkv_bench --op both`。吞吐对
trial 汇总 median/p95/p99；延迟从 workload 前后的 server
`dfkv_op_latency_seconds` histogram **差值**计算 median/p95/p99（不是从终身累计
值猜测）；错误率来自 bench `fails/count`。因此两套目标都必须开启 metrics port，
并在隔离压测窗口执行，避免其它流量混入 histogram delta。

```bash
DFKV_RDMA=1 DFKV_RDMA_DEV=ib7s400p0 DFKV_RDMA_DEPTH=4 \
deploy/dfkv_load_regression.py \
  --baseline-mds 10.0.1.1:9400 --baseline-group glm \
  --baseline-metrics 10.0.1.11:28010,10.0.1.12:28010 \
  --candidate-mds 10.0.2.1:9400 --candidate-group glm \
  --candidate-metrics 10.0.2.11:28010,10.0.2.12:28010 \
  --size 2752512 --count 8000 --threads 64 --batch 1 --bc 1 \
  --warmup-runs 1 --warmup-count 512 --runs 5 \
  --ready-timeout 30 --run-timeout 600 --metrics-timeout 5 \
  --max-throughput-regression 10 --max-latency-regression 20 \
  --max-error-rate 0.01 --output load-regression.json
```

静态环可分别改用 `--baseline-members` / `--candidate-members`；两侧二进制不同时用
`--baseline-bench` / `--candidate-bench`。每次 measured run 使用新 key seed，
避免 write-once cache 把重复 PUT 误算成性能。任一命令超时、metrics reset、无采样
或 baseline 本身错误率越界时 fail closed（退出 `1`）；candidate 越阈值退出 `3`；
通过退出 `0`。无论通过、回归或运行错误，`--output` 都原子写 JSON artifact。

## 6. 回滚（路由可秒级，数据面按 §4e 完整回切）

- RDMA v2 没有协议内降级。切 TCP 仍是 v2 格式，不等于回到旧版本；回退上一发行版
  必须把 client artifact、namespace code 和旧 endpoint/group 作为一个原子单元恢复。
- 只要 §4e 要求的旧服务/目录仍保留，先摘 candidate client 流量，再恢复旧 client
  配置并启动旧服务即可；不得把新旧 endpoint 放进同一 failover 列表。
- SGLang 可将受控副本完整切回原 backend（如 Mooncake）并重启；先校验模型输出，
  再恢复流量。
- 停止使用 candidate 专属 unit（例如 `dfkv-v2-candidate-*`），不要执行模糊的
  `systemctl stop dfkv*`，以免误停共机旧服务。
- 保留 candidate cache 目录、etcd prefix、日志与验收 artifact 供定位；不执行
  `rm -rf` 或 `etcdctl del --prefix`。确认退役后只迁入归档路径。
- 全程不修改 dingo-cache / dingofs / 生产 MDS / 对象存储。

## 7. 监控 / 边界
**完整指标/CLI 参考见 [METRICS.md](METRICS.md)。** 要点：
- **Prometheus 抓取**（opt-in）：`dfkv_server`/`dfkv_mds` 加 `--metrics-port <p>` → `GET /metrics`、`/healthz`、`/readyz`。`dfkv_mds` 的 `/healthz` 是纯进程 liveness（**不探 etcd**：避免 etcd 抖动引发全副本 CrashLoop + lease 集体过期）；`/readyz` 是 TTL-debounced etcd probe（`DFKV_MDS_PROBE_CACHE_MS`，默认 2500 ms，0=逐请求现场 probe），运行期故障时 503、恢复后 200，scheduler 必须用它摘流。cache `/healthz` 反映当前 store/RAM terminal health；`/readyz` 还要求本地启动完成且（配置 MDS 时）**首次 MDS 注册成功**。首次注册在有界 deadline 内失败会退出 1；成功后的 heartbeat 丢失不清 startup-ready latch，须用 registrar heartbeat 指标独立告警。`--id/--group` 成为 `{node,group}` 标签（不设=无标签，向后兼容）。**缺省不开端口 → 行为与旧版一致、对数据面零影响**。Prometheus 直接抓每节点 `:<p>/metrics`。metrics HTTP 监听本身有加固：`DFKV_METRICS_FIRST_REQ_MS`（默认 30000 ms，0=关，硬上限 3600000）给首行请求设 absolute deadline，`DFKV_METRICS_MAX_CONNS`（默认 64，硬上限 4096）限制并发 scrape 连接，超限连接直接关闭；`--metrics-bind <addr>` 可把监听绑到指定地址（缺省 0.0.0.0，`dfkv_server`/`dfkv_mds` 均支持）。
  - 服务端含：put/hit/miss、bytes、淘汰、quota limit/usage/rejection、错误分型、TCP handler limit/timeout/reject、registrar heartbeat/deadline、`open_connections`、per-disk、**采样延迟直方图 `dfkv_op_latency_seconds{op}`**、RDMA 完成/错误/活跃连接。
  - MDS 含：register/keepalive/list/lease/etcd-error、local lease current/pruned + members gauge。
  - 客户端（SGLang 插件 `/metrics`）：`dfkv_client_*{tp_rank}`，见 [CONNECTORS.md](CONNECTORS.md) §2.4 / METRICS.md §3.3。
- **集群/环视图**（CLI，无需开端口）：
  - `dfkvctl ring --mds <eps> --group <g>` — 成员表 + 一致性哈希环每节点 vnode 占比。
  - `dfkvctl stat --all --mds <eps> --group <g>` — 逐节点指标 + 集群聚合（容量/对象/命中率）。
  - `dfkvctl stat <ip:port>` — 单节点原始 Prometheus 文本（旧用法不变）。
- SGLang `--enable-cache-report` 的 HiCache storage hit/miss、TTFT。
- 生产只读：不改现网组件；dfkv 端口（含 `/metrics`）仅内网开放、无鉴权勿暴露公网。
- RDMA 只接受显式 v2 frame；能力 probe、QP 标记或共享 segment 信息不匹配时连接失败，不会把输入按其他版本解码。
