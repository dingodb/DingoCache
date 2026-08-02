# dfkv 独立上线 runbook（与现网 dingo-cache 混部）

> 在 GPU 节点上**独立**起一套 dfkv KV 缓存集群供 SGLang HiCache 用，与现网
> `dingo-cache` 混部互不干扰。**对现网 dingofs/sglang 命名空间、dingo-cache 进程、
> MDS、对象存储一律不动（只读边界）**；dfkv 用独立端口/目录/进程。
> 前提：GLM-5.1 = MLA（每页 KV ≈ 2.74 MiB 单对象、跨 TP 复制、仅 tp_rank0 写）。
> 已在 400G InfiniBand 上端到端验证（两端零拷贝，单口 GET ~93% 线速）。

> **本文只讲 dfkv 集群自身的部署;各推理引擎如何对接/配置 dfkv(HiCache/vLLM/LMCache + 客户端配置总表)见 [docs/CONNECTORS.md](CONNECTORS.md)。**

---

## 0. 网络模型（关键）

dfkv 把**控制面**与**数据面**解耦：

- **控制面 = TCP + 两边 SEND/RECV**：bootstrap TCP 只交换设备/QP/receive-segment 描述；RDMA QP 上每个请求只发送 4-KiB 内的 descriptor、status 和小 header。
- **payload = one-sided RDMA**：v2 PUT 用 `RDMA_WRITE_WITH_IMM` 直落 server 共享 receive-segment slot，GET 由 server `RDMA_WRITE` 到 client 提交的 `{addr,rkey,len}`。数据 fabric 无需 IP。
- **设备发现**：留空时两端各选本地首个 port 1 `ACTIVE` HCA；显式逗号白名单才开启多轨轮转，多 fabric 节点须过滤到两端同名且互通的 fabric。
- **回退**：未设 `DFKV_RDMA` 走 TCP；RDMA client 默认探测 v2，老 server、强制 v1 或共享 segment 暂无 slot 时按连接回退 v1 SEND/RECV。

发现：默认走 **MDS 动态发现**（etcd + dfkv_mds，见 §2b）；静态成员表仍作为遗留/单节点备用路径（见 §4-legacy）。无副本（一致性哈希单属主，节点挂 = 该分片 miss → 重算）。

---

## 1. 构建可移植产物（同架构构建机一次）

> **产物清单**（含 MDS 新增二进制）：`build/dfkv_server`、`build/dfkv_mds`、
> `build/libdfkv.so`、`build/dfkv_smoke`、`build/dfkvctl`、`build/dfkv_bench`。

```bash
git clone git@github.com:ketor/dfkv.git && cd dfkv
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DDFKV_WITH_RDMA=ON -DDFKV_STATIC_LIBSTDCXX=ON      # RDMA + 跨发行版可移植
cmake --build build -j
# 产物：build/dfkv_server  build/dfkv_mds  build/libdfkv.so
#        build/dfkv_smoke  build/dfkvctl  build/dfkv_bench
ldd build/dfkv_server | grep ibverbs                      # 确认链接了 RDMA 库
```
依赖：`libibverbs-dev`（构建期）+ 运行节点装 `rdma-core`。无 RDMA 也可去掉 `-DDFKV_WITH_RDMA` 构建纯 TCP 版。
> QP 信息走 TCP bootstrap 交换（非 librdmacm），所以只依赖 libibverbs，不需要 librdmacm。

> ⚠️ **glibc 下限 = 构建机的 glibc**。在新发行版（如 Ubuntu 24.04 / glibc 2.39）上构建的二进制，拿到老节点（glibc 2.35）会 `version GLIBC_2.3x not found` 起不来。
> **要部署到 glibc 2.35 的节点（如目标 GPU 节点），就在 glibc ≤ 2.35 的环境构建**（Ubuntu 22.04 / RHEL9）。仓库根的 `Dockerfile` 已固定 `ubuntu:22.04` + RDMA + 静态 libstdc++，**一次构建、glibc≥2.35 处处可跑**：
> ```bash
> docker build -t dfkv-build --target build . && id=$(docker create dfkv-build) && docker cp $id:/out ./dist && docker rm $id   # dist/bin/* dist/lib/libdfkv.so
> ```
> `DFKV_STATIC_LIBSTDCXX` 把 libstdc++/libgcc 静态进产物（这些不是 glibc）；libibverbs 无法静态（运行时 dlopen 驱动），运行节点仍需装 rdma-core。

## 2. 每节点：分发 + 缓存目录

```bash
install -m755 build/dfkv_server /usr/local/bin/dfkv_server
install -m755 build/libdfkv.so  /usr/local/lib/ && ldconfig
mkdir -p /mnt/disk1/dfkv /mnt/disk2/dfkv /mnt/disk3/dfkv   # 与现网 dingo-cache 子目录错开
```
容量隔离：`--cap`（总量，按盘均分）自带 LRU 自限；设保守值，确认 `现网用量 + dfkv cap + 预留 < 物理总量`。

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
# --metrics-port: 可选，开 Prometheus /metrics（缺省=不开端口，见 §8 / docs/METRICS.md）
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

> **无状态**：dfkv_mds 自身不持久化任何数据，状态全在 etcd。重启/增减副本无需协调；
> 节点和客户端各持一份 MDS 端点列表，自动故障转移，无需负载均衡器。
> etcd member key 路径：`/dfkv/v1/groups/<group>/members/<id>`；epoch = etcd revision。

## 3. 每节点：systemd unit

`/etc/systemd/system/dfkv.service`：
```ini
[Unit]
Description=dfkv KV cache node (SGLang HiCache L3)
After=network-online.target
[Service]
Type=simple
# v2 shared segment 示例：server 上限 4 MiB、depth=4、2 GiB segment。
# 按所有 rank/process 的 peak live + pooled QP 公式复算，见 CONNECTORS §1.2.1。
Environment=DFKV_RDMA_SERVER_PROTOCOL=auto-v2
Environment=DFKV_RDMA_DEPTH=4
Environment=DFKV_RDMA_RECV_SEGMENT_SIZE=2147483648
# --port = TCP(bootstrap+TCP数据) 端口; --rdma-port = RDMA bootstrap 端口; --rdma-dev = 数据面 400G 口
# --metrics-port = 可选 Prometheus /metrics（缺省=不开端口；--id/--group 成为指标标签）
ExecStart=/usr/local/bin/dfkv_server \
  --dir /mnt/disk1/dfkv,/mnt/disk2/dfkv,/mnt/disk3/dfkv \
  --port 28000 --rdma-port 28001 --rdma-dev ib7s400p0 --max-msg 4194304 \
  --cap 6597069766656 --mds 10.0.0.1:9400,10.0.0.2:9400 \
  --metrics-port 28010 --group default --id n57 \
  --advertise 192.168.1.57:28001
Restart=on-failure
RestartSec=2
CPUQuota=1600%
MemoryMax=64G
Nice=5
LimitNOFILE=1048576
LimitMEMLOCK=infinity        # RDMA 需要锁页内存
[Install]
WantedBy=multi-user.target
```

> `DFKV_RDMA_MAX_BLOCK_BYTES=4194304` 是 **client-only DCP2 声明**，须注入每个
> SGLang/vLLM/LMCache inference client 进程；server 用 `--max-msg 4194304`
> 验证上限，不读取该 client env。

> **可选存储/加速开关（见 [ARCHITECTURE.md](ARCHITECTURE.md) §5–7）。行为开关均为「门面 flag + `DFKV_*` env 双生」，flag 覆盖预设 env；运行时真值经 `dfkvctl ring` INFO 列审计（`engine=`/`wr=`/`ram=`）。**
> - `--store-engine file|slab`（默认 `file`）：`slab` = extent 池 + slots.tbl 重启保温，消除"每块一文件"隐患。**切 slab 需清盘冷启**（旧 blocks/ 布局不复用），属独立迁移动作。
> - `--slab-write direct|buffered`（**默认 `direct`**）：slab 数据面全程 O_DIRECT（写+对齐读+异步 prep），零 page cache/脏页占用——GPU 节点上突发吸收交给显式 RAM 热层而非内核缓存；extent 在 direct 模式下 fallocate 实体化（升级后 df 显示预分配为预期行为）。文件系统拒绝 O_DIRECT（如 tmpfs）时整店回退 buffered，以 `wr=` 上报真值。
> - `--ram-tier on`（默认关）：写直通 RAM 热层 + RDMA arena 零拷贝 GET；`--ram-tier-bytes <bytes>` 定 arena 大小（预注册即 pin 内存，须核 `MemoryMax` 有余量）。开后观测 `dfkv_ram_hit_total` / `dfkv_ram_put_bypass_total`（见 METRICS.md §3.1）。**direct 模式下 flusher 经 CacheDirect DIO 落盘，RAM 池写入量不过 page cache。**
> - 微调项（env only）：`DFKV_SLAB_TABLE_SYNC_MS`（slots.tbl fdatasync 节奏，默认 100，0=关；限定崩溃复活窗口）、`DFKV_RAM_FLUSH_THREADS`（RAM 落盘 worker 数，**默认=4×盘数、上限 16**——小对象落盘是 IOPS 型负载，实测 3 盘节点 64KB 饱和写 3 worker 6.8k → 16 worker 11.7k ops/s；旧"=盘数"默认源自写入路径锁竞争尚未修复的年代）。
> - `--slab-reclaim-ms <n>` / `--ram-reclaim-ms <n>`（env `DFKV_SLAB_RECLAIM_MS` / `DFKV_RAM_RECLAIM_MS`；默认 50 / 10，`0`=关）：后台空槽回收节奏。回收线程按需求水位（上个周期插入量的 2 倍，上限 = 该 class 容量的 1/4）小批量预驱逐，使满店/满 arena 下 PUT 的分配走"摘现成空槽"快路径，不再内联跑 CLOCK 扫描（满 5 TiB 店实测：冷数据集 PUT 6.0→43.2 GB/s、稳态 12.7→29.9 GB/s、class 迁移期 p99 501→73 ms）。**默认值即推荐值，正常无需调**：池有未绑 extent 时不驱逐（Put 免费绑新 extent，驻留无损）、flush 积压深时 RAM 侧自动歇拍（那时瓶颈在落盘不在空槽），所以开着不伤命中率；仅在排查回收线程本身时才设 0 关闭对照。观测：`dfkv_slab_reclaimed_total` / `dfkv_ram_reclaimed_total`（见 METRICS.md）。
>   回收线程同时承担**类再平衡**：满店/满 arena 上出现新 value 尺寸（换模型/改 page 几何）时，热 class 按需从最冷 donor class（本周期零写入或需求 <1/4，且容量高于 8-extent 条带下限=类配额保护）搬 extent 扩容，而不是像内联路径那样"自噬"（新类永远长不大、写后读回大量 miss）。每 tick 限速（盘 2 / RAM 8 个 extent），秒级收敛；观测 `dfkv_slab_rebalanced_total` / `dfkv_ram_rebalanced_total`。
> - `--slab-granularity <bytes>`（默认 1 MiB）：slot 量子，小值负载调小（64KB 值在 1MiB 粒度下浪费 94%）。**改动 = 布局变化 = 该店清空冷启**，按迁移动作对待。
> - `--put-inflight-limit <n>`（默认 0=关）：并发盘写超过 n 的 PUT 以 kCacheFull 快速拒绝（客户端视为普通 put 失败、不进 cooldown）= 用受控 miss 换掉过载排队尾延迟。RDMA 与 TCP 两条数据路径同受此门约束；RAM 热层的异步 flusher 落盘**不受**此门限制（否则背压会放大为 flush 丢弃）。
> - RAM 热层 arena 预触 + RDMA MR 注册都在启动期走页：**arena 每 16 GiB 约 +5-10s 启动时间**，配大 arena（≥64 GiB）时同步调大 systemd `TimeoutStartSec`（默认 90s）并让就绪探测等待 metrics 端口。
> GPU 节点推荐组合：`--store-engine slab`（direct 已默认）+ `--ram-tier on` + 按节点突发画像定 `--ram-tier-bytes`。
```bash
systemctl daemon-reload && systemctl enable --now dfkv
journalctl -u dfkv -n 10 --no-pager
# 应见 "PORT 28000" + "RDMA listening (TCP bootstrap) on port 28001, dev=ib7s400p0"
#      + "dfkv_server registered with MDS group=default id=n57 advertise=192.168.1.57:28001"
```
> server 的 bootstrap 监听 `0.0.0.0`，靠防火墙限制在内网。优雅关闭已修（`systemctl stop` 约 1s 退出）。
> **多轨**：server 启动时 anchor 白名单内全部 active rail；client 新 QP 轮转健康轨。留空时两端各选本地首个 ACTIVE HCA，适合 local device 名不同的单 fabric 主机；显式 `--rdma-dev` / `DFKV_RDMA_DEV` 会把设备选择发给 peer，故生产多 fabric 主机必须在两端过滤到**同名且互通**的一组设备。`DFKV_RDMA_NUMA=1` 只改变 QP 选轨与 serve 线程亲和；当前 receive segment 是单块 process-wide 分配，不是 per-NUMA/per-rail 分片。
>
> **v2 segment 预算**：`slot=align4K(4096 + sizeof(ValueHeader) + max(max_block,4096))`；
> `segment >= Σ(live + client-pool-idle data/control QP × depth × slot)`。lease
> 保留到 QP 销毁或 idle reclaim，不能只数在飞请求。4 MiB/depth=4/2 GiB
> 约容纳 127 条 data QP（未扣 control lease）。上线先看
> `dfkv_rdma_recv_segment_free_bytes`、`dfkv_rdma_v2_ready`、
> `dfkv_rdma_{v1,v2}_conns_opened_total`；free 接近 0 会让新连接回退 v1。

## 4. 集群成员管理

### 4a. MDS 动态发现（推荐）

节点通过 `--mds` + `--group` + `--id` + `--advertise` 向 MDS 自注册，客户端调用
`dfkv_start_mds_discovery(c, "mds1:9400,mds2:9400", "default", 3000)` 周期轮询
MDS（默认间隔 3000 ms）。epoch（etcd revision）变化时自动重建加权 Ketama 环。
增减节点只需启停 `dfkv_server`，无需改客户端配置。

两层离线检测：
- **层 2（权威）**：etcd lease 到期 → MDS 视图变更 → 客户端 epoch 推进 → 环重建（≤ 30 s）
- **层 1（快速）**：`PeerHealth` 传输 IO 失败即短路该节点为 miss 并进入 cooldown，不触发环重建

### 4b. 静态成员表（遗留/单节点备用）<a name="4-legacy"></a>

适用于无 etcd 的简单环境或单节点调试。成员 = 所有节点的
`name=<bootstrap-ip>:<rdma-port>`（RDMA）或 `:<port>`（TCP）：
```
n57=192.168.1.57:28001,n58=192.168.1.58:28001,...
```
增减节点 = 改成员字符串并重载 SGLang 端。`dfkv_server` 此时不带 `--mds` 启动。
建议 N≥4 降低单点 miss 影响。

## 5. 上线顺序 + 冒烟（无需 GPU）

1. （MDS 路径）起 etcd，再起所有 `dfkv_mds` 副本（§2b），确认日志 "listening"。
2. 所有节点起 `dfkv_server`（§3），确认 PORT + "registered with MDS" 日志。
3. 冒烟（任一能访问内网的机器）：
   ```bash
   dfkv_smoke --members n57=192.168.1.57:28000 --size 2752512                          # TCP
   DFKV_RDMA=1 DFKV_RDMA_PROTOCOL=auto-v2 DFKV_RDMA_MAX_BLOCK_BYTES=4194304 \
     DFKV_RDMA_DEV=ib7s400p0 dfkv_smoke --members n57=192.168.1.57:28001 --size 2752512
   ```
4. 端到端零拷贝校验（插件 → libdfkv → RDMA → server，验证 payload 直落缓冲）：
   ```bash
   DFKV_RDMA=1 DFKV_RDMA_DEV=ib7s400p0 DFKV_MEMBERS='n=192.168.1.57:28001' \
     python3 test/python/rdma_e2e_validate.py    # 期望 RESULT: ZERO-COPY RDMA E2E OK
   # 随后在 server /metrics 确认 v2_ready=1、v2_conns_opened_total 增长且 free_bytes 有余量
   ```
5. 压测（可选）：`DFKV_RDMA=1 DFKV_RDMA_DEV=ib7s400p0 dfkv_bench --members ... --size 2752512 --count 8000 --threads 64`。
6. 在**一个受控 SGLang 副本**上切 `dynamic` 后端，发共享长前缀请求看命中上涨，确认后推广。

## 6. 回滚（秒级）
- 仅回滚 RDMA v2（保留 dfkv）：在**每个推理 client 进程**设置 `DFKV_RDMA_PROTOCOL=1` 并重启；需要 server 同时禁用时，在**每个 dfkv server 进程**设置 `DFKV_RDMA_SERVER_PROTOCOL=1` 并重启。变量只在 transport/server 构造时读取，已有连接池 QP 不会重新协商；恢复自动探测时也必须删除对应变量并再次重启所有受影响的 client/server。每次切换后用 v1/v2 connection counters 验证实际协议。
- SGLang：`--hicache-storage-backend` 改回原后端（mooncake 等）重启该副本，与 dfkv 解耦。
- dfkv 节点：`systemctl stop dfkv`；缓存可丢（KV 可重算），彻底清理删 `/mnt/diskX/dfkv`。
- dfkv MDS：`systemctl stop dfkv_mds`；无状态，etcd 数据可保留也可清除（`etcdctl del /dfkv --prefix`）。
- 全程不影响 dingo-cache / dingofs / 生产 MDS / 对象存储。

## 7. 监控 / 边界
**完整指标/CLI 参考见 [METRICS.md](METRICS.md)。** 要点：
- **Prometheus 抓取**（opt-in）：`dfkv_server`/`dfkv_mds` 加 `--metrics-port <p>` → `GET /metrics`、`/healthz`。`--id/--group` 成为 `{node,group}` 标签（不设=无标签，向后兼容）。**缺省不开端口 → 行为与旧版一致、对数据面零影响**。Prometheus 直接抓每节点 `:<p>/metrics`。
  - 服务端含：put/hit/miss、bytes、淘汰、错误分型、`open_connections`、per-disk、**采样延迟直方图 `dfkv_op_latency_seconds{op}`**、RDMA 完成/错误/活跃连接。
  - MDS 含：register/keepalive/list/lease/etcd-error + members gauge。
  - 客户端（SGLang 插件 `/metrics`）：`dfkv_client_*{tp_rank}`，见 [CONNECTORS.md](CONNECTORS.md) §2.4 / METRICS.md §3.3。
- **集群/环视图**（CLI，无需开端口）：
  - `dfkvctl ring --mds <eps> --group <g>` — 成员表 + 一致性哈希环每节点 vnode 占比。
  - `dfkvctl stat --all --mds <eps> --group <g>` — 逐节点指标 + 集群聚合（容量/对象/命中率）。
  - `dfkvctl stat <ip:port>` — 单节点原始 Prometheus 文本（旧用法不变）。
- SGLang `--enable-cache-report` 的 HiCache storage hit/miss、TTFT。
- 生产只读：不改现网组件；dfkv 端口（含 `/metrics`）仅内网开放、无鉴权勿暴露公网。
- RDMA v1/v2 frame 均带显式版本；新 client 先做能力 probe，滚动升级中的新旧 client/server 组合自动选择共同的 v1，不会把另一版本静默错读。`dfkv_rdma_{v1,v2}_conns_opened_total` 可审计实际路径。
