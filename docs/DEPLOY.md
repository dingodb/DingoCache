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
- **payload = one-sided RDMA**：inline PUT 用 `RDMA_WRITE_WITH_IMM`，大对象通过操作级 lease 接收；GET 一律由 client RDMA READ 拉取 exact grant，等待 RELEASE ACK 后连接回池，不保留固定 pull arena。server 只兼容 v2.28 默认 dynamic 握手，不接受旧 WRITE GET。数据 fabric 无需 IP。
- **设备发现与拓扑**：留空时自动发现保持 `ACTIVE`-only（两端各选本地首个 port 1 `ACTIVE` HCA）；显式逗号白名单定义固定 topology，按首次出现顺序去重，server 会接纳**存在但启动时 DOWN** 的设备完成 anchor/MR 初始化并持续监控。运行期至少一条 initialized rail 健康即可承载 placement；最后一条健康 rail 丢失立即退环，0→非 0 恢复仍通过连续采样门。
- **失败策略**：未设 `DFKV_RDMA` 时选择 TCP；一旦选择 RDMA，显式设备缺失、open/port/GID query 失败，或任一 configured rail 的 anchor、启动时的试注册（初始 receive chunk 注册一次随即释放，`LimitMEMLOCK` 不够时在这里失败）、RAM/user MR 初始化失败，均拒绝启动而不缩小 topology。client 配置 `DFKV_RDMA_RAIL_TIERS` 后还要求 MDS 返回完整 HLT1 peer topology；缺失/不完整或无兼容健康轨以 client-local `kNoCompatibleRail` fail closed，不切 TCP、不惩罚 peer。

- **信任模型（RDMA 数据面）**：server 不鉴别 client，能连到 bootstrap 端口的进程都能建连；隔离的目标是一个出错的 client 不能破坏别的连接，不是防恶意 client。每条连接只拿到覆盖**自己 receive slots** 的 rkey（按连接单独注册，不再是整块 chunk 的 rkey），写别的连接的 slot 会以远端访问错误失败；slot 交给新连接前清零，读不回上一个连接留下的字节；leased PUT 与 GET 用按操作注册、只覆盖本次范围的 MR，用完即撤销。仍未覆盖的：同一设备上所有 MR 共用一个 PD，猜中别的连接 rkey 的恶意 client 可以读或写那块范围，包括仍在有效期内的按操作 READ 授权（GET 的 staging、钉住的 RAM 值），即不仅能破坏、也能读到别人的值；多窗口 PUT 的窗口位置无法验证，覆盖不全时可能存下自己 lease 里已有的字节（staging 不逐次清零）。要防恶意租户需要按 QP 绑定的 memory window，目前未做。

发现：默认走 **MDS 动态发现**（etcd + dfkv_mds，见 §2b）；静态成员表仍作为遗留/单节点备用路径（见 §4-legacy）。无副本（一致性哈希单属主，节点挂 = 该分片 miss → 重算）。

### v2.31 过渡升级

先升级 server，再升级推理 client。server 过渡层接受 v2.28 默认 dynamic
client；显式关闭 dynamic 的旧连接不兼容。新 client 只支持新的 token-free
握手，不能提前部署到旧 server，也不能升级 client 后单独回滚 server。
逐副本排空请求并重启完整 TP/PP/EP 副本，核对每个 rank 的实际库版本；
磁盘铺货或切换软链不会更新运行中的进程，不做进程内 `.so` 热替换。
节点重启时的重连、外部命中恢复和模型请求仍需单独验收。


---

## 1. 构建可移植 release 产物（同架构构建机一次）

> v2.11.1 的 canonical Linux 交付是一个自包含 tarball，含 versioned
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
依赖：`libibverbs-dev`（构建期）；运行节点须安装 libibverbs 及对应硬件 provider。Ubuntu/Debian 运行包为 `rdma-core libibverbs1 ibverbs-providers`，只装前两项不足以发现设备。发布的 runtime 镜像包含这三项；设备透传不能替代用户态 provider。无 RDMA 也可去掉 `-DDFKV_WITH_RDMA` 构建纯 TCP 版。
> QP 信息走 TCP bootstrap 交换（非 librdmacm），所以只依赖 libibverbs，不需要 librdmacm。
> CacheLib/Navy 不是 v2.0.0 的构建依赖，也没有占位 backend/CMake 开关；
> 评估结论、兼容性差距和重开条件见
> [CACHELIB_EVALUATION.md](CACHELIB_EVALUATION.md)。


> ⚠️ **glibc 下限 = 构建机的 glibc**。在 Ubuntu 24.04 / glibc 2.39
> 直接构建的二进制不能部署到更旧的 glibc。发布产物要求 **glibc ≥ 2.34**（可在
> RHEL 9 系 2.34 主机上运行）：CI 的 portable job 和 release workflow 拒绝任何需要
> GLIBC_2.35 及以上符号的二进制。CI 的 portable job 和仓库根
> `Dockerfile` 都固定 Ubuntu 22.04 + RDMA + 静态 libstdc++，并构建同一
> canonical tarball。`DFKV_STATIC_LIBSTDCXX` 只静态链接 libstdc++/libgcc；
> libibverbs 仍动态加载 provider，运行节点必须同时具备对应插件。镜像验收要在可用 RDMA 设备上完成真实 PUT/GET；仅 `--version` 或 TCP 冒烟不能证明 RDMA 可用。

> 发布二进制带加固编译选项（`DFKV_HARDENING`，默认开：`-fstack-protector-strong`、
> `-D_FORTIFY_SOURCE=2`（编译器已自带更高或相同级别时沿用它）、`-z relro -z now`；Debug 构建不加 fortify，sanitizer 构建关掉），
> 并已 strip。调试符号在单独的 `dfkv-<版本>-linux-x86_64-debug.tar.gz`：目录结构
> 与发布包相同（`bin/<工具>.debug`、`lib/libdfkv.so.<版本>.debug`），解压到发布包
> 同一目录后 gdb 按二进制里的 `.gnu_debuglink` 自动找到；分析 core 时才需要，
> 不必部署到节点。它约 66 MB（2.42 实测；发布包约 10.6 MB），因为每个工具都静态链接了
> 整个 core 库，各自带一份调试信息（`dfkv_server.debug` 约 25 MB）。

## 2. 每节点：分发 + 缓存目录

```bash
version=2.11.1
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

**容量和盘不能在线调整。** `--cap` 与 `--dir` 列表决定每盘容量，slab 把它写进
每盘的几何（`slab_meta`），几何不符的目录拒绝启动。改 `--cap`、增删盘都要停服务、
清空全部缓存目录后冷启动（缓存内容全部重来）。换一块坏盘不在此列：目录列表和
`--cap` 保持不变，把新盘挂到原路径（空目录），重启后只有这块盘是空的，其余盘的
数据保留。

## 2b. MDS 层：etcd + dfkv_mds

> **推荐路径**：用 MDS 动态发现时须先起 etcd 集群和 dfkv_mds 副本。

### 2b-1. etcd 集群

使用现有 etcd 或按官方文档起 1/3 节点集群。心跳只续租约、读一次记录，不写
etcd；记录只在内容变化时重写，仅统计（STA1）变化时最多每 `DFKV_MDS_STATS_PUT_MS`
（默认 60 s）重写一次，所以每个节点每分钟约 1 个 etcd revision（2.42 及之前每次
心跳一次 Put，3 个 MDS 端点轮转时还要每次重新授予租约，每分钟 6 个）。
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
# N 个连接最多占 2N+128 个 fd（每个 handler 处理请求时还要连一次 etcd），默认
# DFKV_MDS_MAX_CONNS=4096 需要 8320。systemd 默认软上限通常只有 1024；dfkv_mds 启动时
# 会把软上限提到硬上限以内够用的值，硬上限仍不够则下调连接上限并打 WARN。这里显式给足。
LimitNOFILE=65536
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
> 读记录 → 续租（租约已过期而失败）→ lease grant → Put，共 4×2s；开了
> `DFKV_MDS_FENCING` 且 fence 名单的缓存（1s）过期时还要先读一次名单，共 5×2s。
> node/client registrar 的默认 socket I/O timeout 为 11s，组合上覆盖这 10s server
> budget；不要把 caller timeout 调到单个 etcd timeout。2.35–2.42 的 registrar 仍是 7s：只在 etcd
> 变慢（每个请求接近 2s）时才可能在这条 4 个请求的路径上超时，下一次心跳重试。
> registrar 粘在一个 MDS 上（起始的那个按 member id 的 hash 选，各节点分散在各 MDS
> 上），失败时在同一次心跳里立即依次试其他端点（建连超时 2s），失败的端点 30s 内
> 不再先选。任一 MDS 都续记录上挂的那个租约，所以换 MDS
> 不再重新授予租约、重写记录。
> MDS 对 member list 的回答复用 `DFKV_MDS_LIST_CACHE_MS`（默认 500 ms，上限 10000；`0`=每次
> 轮询都读 etcd，2.42 及之前的行为），同一 group 并发的轮询只读一次 etcd；本 MDS
> 写入的记录立即可见，其他 MDS 写入的最多晚这么久。client list（`dfkvctl clients`、
> 冷却改道的登记门）用同一个值，但最多 1000 ms（低于客户端的轮询间隔）。
> **摘除节点**：没有 fencing 时要三步（改环、停进程、`systemctl disable`，否则
> 进程或 systemd 拉起后 10s 内又注册回来）。同一 group 的全部 MDS 设
> `DFKV_MDS_FENCING=1` 后一步即可：`dfkvctl member fence <id> --mds <eps> --group
> <g>` 写一条无租约的 fence 记录并删掉该节点的 member 记录，客户端约 6s 内（3s
> 轮询一次，环的视图要连续两次一致才采纳）就不再路由到它；节点进程可以继续运行，它的注册和心跳被拒（节点日志
> `is fenced`、指标 `dfkv_mds_fenced 1`），直到 `dfkvctl member unfence <id>`，之后
> 下一次心跳重新注册。`dfkvctl member fenced` 列出当前被 fence 的节点。
> fence 和 unfence 与注册一样没有鉴权：能连到 MDS 端口的人都能执行，信任模型与
> 注册相同（控制面端口只对集群内开放）。各 MDS 的 fence 名单最多缓存 1s：fence 之后
> 1s 内经别的 MDS 到达的心跳可能把记录写回，但 member list 照样过滤掉它，记录在约
> 30s 后随租约过期消失。
> MDS 的 `/healthz` 是纯进程 liveness（不 probe etcd）：否则一次秒级 etcd
> 抖动会让 kubelet 同时重启全部 MDS 副本，CrashLoop 比抖动更久，重注册风暴
> 还会引发 lease 集体过期。`/readyz` 保留 etcd 依赖检查（活着但探不到 etcd
> 的 MDS 无法服务 membership，必须摘出路由），但经 TTL-debounced probe
> （`DFKV_MDS_PROBE_CACHE_MS`，默认 2500 ms；0=恢复逐请求现场 probe；钳制
> 上限 600000 ms），kubelet 抓取节奏不会放大成 etcd 读负载，not-ready 副本
> 每 TTL 最多重探一次。运行期 etcd 不可达时 `/readyz` 503 使 scheduler
> 摘流，etcd 恢复后同一进程自动回 200；`dfkv_mds` 不因运行期 etcd 短暂故障
> 退出。

> **etcd 拒绝写入时的 member list**：etcd 空间配额用尽后只接受读和删除，Put 与
> lease grant 被拒，已有 lease 照常到期。节点心跳续不上，member key 会在节点
> 完全健康的情况下从 etcd 消失；registrar 在 3 个及以上 MDS 端点间轮转时，全部
> member 在一个 TTL 内消失。MDS 会记住“heartbeat 已到达本 MDS、但没能写进
> etcd”的 member：`ListMembers`/`ListTopology` 读到的 etcd 视图缺少这样的
> member 时返回 `kIOError`，不把残缺或空的视图当作成员表下发。只记 heartbeat、
> 不记 register：registrar 只在本进程注册成功之后才发 heartbeat，所以
> heartbeat 一定来自曾在成员表里的节点；注册一直失败的节点（新节点，或
> key 过期后才回来的节点）本进程尚未注册成功，不会让 list 被扣住。各版本客户端在
> list 失败时都保留 last-good ring，所以**全部** MDS 副本升级后现网客户端即受
> 保护；未升级的副本仍会下发残缺视图。
> - 恢复：这些 member 全部回到 etcd（由本 MDS 或其他 MDS 写入均可），或已有
>   `DFKV_MDS_UNWRITTEN_HOLD_MS`（默认 120000 ms = 4×TTL，钳制上限 600000；
>   `0`=关闭，恢复 2.36 及之前的行为）没再向本 MDS 心跳。该窗口必须大于
>   “MDS 端点数 × 10s 心跳间隔”（registrar 轮转一圈的时间），否则记录会在节点
>   回到本 MDS 之前失效。etcd 恢复写入后各节点在下一次心跳（10s 内）重新写入，
>   无需重启 MDS、节点或客户端。心跳在某个 MDS 上失败之后真正下线的节点，会让
>   该 MDS 的 list 最多再扣一个窗口，之后照常移除。
> - 不受影响：心跳写失败但 etcd 仍列出该 member（lease 尚未到期）时 list 照常
>   下发；节点正常下线、其他 group、`ListClients` 不变。
> - 期间的表现：新启动的客户端拿不到 ring（全部按 miss 运行），`dfkvctl
>   ring/topology/stats` 报 `MDS query failed`；节点侧心跳失败见
>   `dfkv_mds_heartbeat_healthy`。
> - 观测：`dfkv_mds_list_withheld_total`、`dfkv_mds_unwritten_members`
>   （docs/METRICS.md §3.2），MDS 日志 `withholding the member list of
>   group=...`（至多每 30s 一条），告警 `DfkvMdsMemberListWithheld`（list
>   连续被扣约 4 分钟后触发；单个节点下线引起的至多一个窗口的扣留不触发）。
> - 局限：
>   - MDS 只了解向它心跳过的 member。刚启动、尚未收到心跳的 MDS，以及节点
>     registrar 端点列表里没有的 MDS，没有记录可依，照常下发 etcd 视图。
>   - 故障期间重启的节点只会发 register，不被记录；一个 group 的节点若在故障
>     期间全部重启，该 group 仍会被下发为空视图。etcd 拒绝写入期间不要重启
>     cache 节点。
>   - MDS 端点为 4 个及以上时，最后一次写入某 member 的那台 MDS 要等该 member
>     的 key 过期之后才再次收到它的心跳（超过 3 个后每多一个端点约晚 10s），这段时间它
>     可能下发缺少该 member 的 list；恰好轮询到它的客户端会采纳略小的 ring，
>     直到 list 恢复下发。

> **混合代际滚动（v1.x ↔ v2）**：`DFKV_MDS_ACCEPT_LEGACY=1` 把 MDS 控制面版本闸
> 按已知 epoch 放宽一档——v1.x 节点/客户端以旧 42/10 字节帧直接被服务（操作码、
> MemberInfo 载荷、etcd schema、租约语义跨代一致，status 字节按 v1 枚举顺序回声）。
> 默认关闭 = 历史严格行为（旧 epoch 帧立即拒连）。这**只打通 MDS 控制面**：数据面
> 仍严格 epoch 6（TCP）/8（RDMA），v1/v2 节点与客户端必须各属一个 group（环级隔离），客户端版本
> 跟随自己的环。迁移期配好后用 `dfkv_mds_legacy_frames_total` 观察旧流量，归零后
> 撤掉该 env 恢复严格模式。


## 3. 每节点：systemd unit
如启用示例 quota 文件，先创建（已存在时绝不覆盖）：
```bash
install -d -m0755 /etc/dfkv
test -e /etc/dfkv/tenant-quotas ||
  install -m0644 /dev/null /etc/dfkv/tenant-quotas
```


`/etc/systemd/system/dfkv.service`（参考**一台生产节点的配置**映射到 v2.0.0 大生产参数：**v1.37 生产开了什么现阶段保留什么**；**v2 才落地的能力（SYS_URING/CONVoy 合并/监听加固/slab 水位线）设置为 drop-in 注释默认留**。更小形（单 HCA、无 RAM tier）只需对 `--rdma-dev`、`--ram-tier`/`--ram-tier-bytes` 做减法）：

> **生产配置 (v1.37-1.40) → v2.0.0 映射关系**：
> | 项 | 生产 (v1.37/1.40) | 本示例 (v2.0.0) | 变动 |
> |---|---|---|---|
> | `--rdma-dev` | 本机全部数据面 rail | 本机全部数据面 rail | 一致 |
> | RDMA depth | `DFKV_RDMA_DEPTH=32` | `DFKV_RDMA_DEPTH=4` | 新默认 4 对齐 client/server——depth=1 会 clamp 批处理窗口，pipeline GET 退化 3-4×、突发 PUT 批失败致 L3 prefetch miss（热轮吞吐 -29.8%，depth=4 后仅 -2.9%）；depth>1 膨胀每连接 segment lease（depth×slot），过大在高并发时易耗尽共享 segment |
> | RDMA_NUMA | `1` | 保留 `DFKV_RDMA_NUMA=1` | 一致 |
> | RAM tier | on | on（`--ram-tier-bytes <size>`） | 一致 |
> | SERVER_URING | 开，depth=32 | 默认关，经 drop-in 打开（须编 `-DDFKV_WITH_URING`） | v2 保留 |
> | READ_COALESCE | 开 | 默认关，经 drop-in 打开（TCP 端共读 convoy） | v2 新增 |
> | 监听加固 | — | `DFKV_TCP_FIRST_REQ_MS=30000` + `DFKV_METRICS_MAX_CONNS=64` | PR#240 新增 |
> | slab 驱逐水位 | — | `DFKV_SLAB_EVICT_HIGH_PCT=92` / `LOW_PCT=88` | v2 新增（默认 92/88，显式解释为风控用途） |
> | systemd Timeout | `TimeoutStartSec=600` / `TimeoutStopSec=600` | 同上 | **大 arena 的预触页在启动时占时间**（4K 页时实测 192 GiB 约 39–47 秒，启动到就绪约 54 秒；拿到透明大页时只要几秒，但大页是尽力而为的，超时仍按 4K 页的耗时来定，并留余量），systemd 默认 90s 对更大的 arena 不够，要与 arena 大小同步扩大。arena 不再整块注册到各 rail，启动和停止各少一段随 arena 大小和 rail 数增长的时间 |
> | OOM | `OOMScoreAdjust=-500` | 保留 | 一致 |
> | tenant quota | `DFKV_TENANT_QUOTAS_FILE` + default=0 | 保留 | 一致 |
```ini
[Unit]
Description=dfkv KV cache node (SGLang HiCache L3)
# 启动顺序：在 MDS 之前动 After；v2 首注册 deadline 60s，必须先起 MDS 再动 server
After=network-online.target dfkv-mds.service
Wants=network-online.target
[Service]
Type=simple
# v2 shared segment：RDMA depth=4、4 MiB block 时每条 pull data QP
# 占 33,587,200 B（receive + source arenas）。所需大小 ≈ 每 server 的 data 连接数
# （client 进程数 × 每 pool 上限）× 2 个 pool × 33,587,200 B × 1.25（churn）；
# 代码通用默认是 128 GiB；这里按实测容量包络显式收紧为 64 GiB。
Environment=DFKV_RDMA_DEPTH=4
Environment=DFKV_RDMA_RECV_SEGMENT_SIZE=68719476736
# 全轨白名单 + NUMA：列出本机全部数据面 HCA（示例为两条 mlx5_0、mlx5_1）。
# 显式白名单是固定 topology：保留首次出现顺序；present DOWN rail 仍必须完成
# anchor/共享 segment/所有配置 MR 初始化，随后由 health monitor 等待 LinkUp。
# client 侧另行注入本机 `DFKV_RDMA_DEV`；GPU/CPU host 可有不同列表。
# 若设置 `DFKV_RDMA_RAIL_TIERS`，名称须来自该 local list，peer HLT1 用同名健康
# 交集决定最高可用 tier。NUMA/credits/latency 只在 tier 内选择，拥塞不降 tier。
# `--rdma-dev` 只配置 server，胜过同名 env（CONNECTORS §1.2）。
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
# --port = TCP(bootstrap+TCP数据) 端口; --rdma-port = RDMA bootstrap 端口; --rdma-dev = 数据面 RDMA 设备
# --metrics-port = 可选 Prometheus /metrics（缺省=不开端口；--id/--group 成为指标标签）;
#   --metrics-bind <addr> 绑定监听地址（缺省 0.0.0.0）
# 此生产示例故意不设置 --store-engine/DFKV_STORE_ENGINE：resolved 默认即 slab。
# 两个 <size> 按本机填十进制字节数：
#   --ram-tier-bytes：能长期锁给 dfkv 的内存（启动时整块预触页并锁住），扣掉系统、
#     推理进程和其他服务需要的部分，并与下面的 MemoryMax、TimeoutStartSec 一起调；
#   --cap：全部 --dir 的总容量（按盘均分），取各盘文件系统可用空间之和，减去每盘
#     slots.tbl（约为该盘容量的 1/16384）和运维预留。
ExecStart=/usr/local/bin/dfkv_server \
  --dir /mnt/disk1/dfkv,/mnt/disk2/dfkv,/mnt/disk3/dfkv \
  --port 28000 --rdma-port 28001 \
  --rdma-dev mlx5_0,mlx5_1 \
  --rdma-depth 4 --rdma-numa 1 \
  --ram-tier on --ram-tier-bytes <size> --ram-tier-shards 16 \
  --cap <size> --mds 10.0.0.1:9400,10.0.0.2:9400 \
  --metrics-port 28010 --group default --id n57 \
  --advertise 192.168.1.57:28001 \
  --mds-registration-timeout-ms 60000
# v2.0.0 生产扩展能力（默认关，会自动在 systemd drop-in 覆盖）：
# - io_uring async GET/direct read（生产为 on;
#   需要编 `-DDFKV_WITH_URING`）：
#Environment=DFKV_SERVER_URING=1
#Environment=DFKV_SERVER_URING_DEPTH=32
# - TCP 端共读 convoy 合并：server 同 key 复本位于重复读共享一个 disk read
#Environment=DFKV_READ_COALESCE=1
#Environment=DFKV_READ_COALESCE_RECUR_MS=1000
#Environment=DFKV_READ_COALESCE_TIMEOUT_MS=500
Restart=on-failure
RestartSec=2
# 大 arena RAM tier 启动时要预触页：4K 页时实测 192 GiB 约 39–47 秒（每 16 GiB 约 4 秒）；
# 拿到透明大页时只要几秒，但大页拿不拿得到看内存状况，超时按 4K 页的耗时来定。
# 生产 TimeoutStartSec/TimeoutStopSec=600，90s 默认会 OOM-timeout 掉 dfkv；
# 配大 arena（≥64 GiB）时必须同步调大。
TimeoutStartSec=600
TimeoutStopSec=600
# 停止时给 RAM 层积压刷盘的时限（默认 30 秒，超时丢弃剩余）。停止超时放到 600 秒之后，
# 这里跟着放大，留出停请求和各盘同步的时间。
Environment=DFKV_RAM_SHUTDOWN_DRAIN_TIMEOUT_S=300
# OOM 调整：内存压力下 Linux OOM Killer 倾向杀占内存过多的进程，dfkv 的 RAM arena 会被白锁。
# 生产取 -500，降低 dfkv 被错杀风险（NVMe cache-process 死了 模型自己还能重算，预 pin 大 RAM 的
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

> **`--max-msg` 默认 32 MiB，是 server 接受的逻辑 block 上限；client
> `DFKV_RDMA_MAX_BLOCK_BYTES` 不得超过它。** client 每次 data Acquire 按当前
> operation 最大对象选择 power-of-two connection class；逻辑上限不再决定所有
> QP 的物理 slot。server 小于 client 逻辑声明仍会拒绝连接，避免超限对象被误收：
> 凡是按对象大小声明连接的路径都是如此，即 inline 阈值
> （`DFKV_RDMA_INLINE_PUT_MAX_BYTES`，默认 4 MiB）设为 0 或不低于 `--max-msg` 时，
> 以及一次操作失败后用新连接重试时（重试不申请 leased PUT）；TCP 路径上 server 对
> 超限请求直接断开连接，client 冷却这个节点。
>
> 只有 RDMA 的 leased 路径不同：在默认 inline 阈值下，超过阈值的 PUT 走 leased
> 路径，连接的 class 不超过阈值，所以 client 上限高于 server 的 `--max-msg` 时连接
> 照常建立，超限的对象由 server 逐个答 `kInvalid`（计入
> `dfkv_rdma_leaseput_oversize_rejects_total`，首次和每 1024 次记一条带对端地址的
> WARN），client 当作未缓存，不冷却节点。2.42.0 及更早的 server 收到这样的请求会
> 直接断开连接而不回复，client 等满 `DFKV_RDMA_OP_TIMEOUT_MS`（设了
> `DFKV_RDMA_BATCH_OP_TIMEOUT_MS` 的批量操作则是它；默认 5 s）后记 completion
> timeout 并冷却这个节点。所以无论哪个版本，client 的上限都应不高于 server 的
> `--max-msg`。
>
> QP 的传输层重传（client 和 server 的 QP 都是）：`DFKV_RDMA_QP_TIMEOUT`（1–31，默认 14）、`DFKV_RDMA_QP_RETRY_CNT`（0–7，默认 7）、`DFKV_RDMA_QP_RNR_RETRY`（0–7，默认 7 即不限）。实测（ConnectX，对端 QP 已销毁）一个 SEND 到 `RETRY_EXC`（status=12）要 3.5–3.7 s，timeout 取 12、14、16 几乎没有差别（网卡有自己的最小 ack 超时），主要由重试次数决定；它小于下面的单操作超时。
> Completion timeout：`DFKV_RDMA_OP_TIMEOUT_MS`（默认 5000）约束单操作；
> `DFKV_RDMA_BATCH_OP_TIMEOUT_MS`（unset/0 = 跟随前者）覆盖 **所有** multi-item
> Cache/Range/Exist、zero-copy 与 SG 窗口。一个窗口共用一个绝对 deadline，
> 部分 CQ completion 不会续期。调用方临时 buffer 在方法返回后即可释放；
> pool 外 MR 已在成功 completion 后解除，失败路径先销毁 QP 再解除。


> **存储/加速开关（见 [ARCHITECTURE.md](ARCHITECTURE.md) §5–7）。解析顺序为 flag > 环境变量 > `slab`；运行时真值经 `dfkvctl ring` INFO 列（`engine=`/`wr=`/`ram=`）和 `dfkv_build_info{engine,write_mode}` 审计。**
> - `--store-engine slab|file`：不设置 flag/env 时所有 store/server 路径默认 `slab`。slab = extent 池 + sparse `slots.tbl` + dirty/clean epoch。**早期格式→v3 tenant-scoped slab 必须使用空缓存目录**；容量、格式或几何不符会拒绝启动而不会改写原数据，也绝不会静默改用 file。`file` 的 48 字符 tenant+object 文件名同样不读取旧 cache，仅作显式诊断/回滚。
> - **slab 首次初始化中断**：新版从空目录建仓前落盘 `slab_initializing`。
>   仅当该标记与请求几何一致、`slab_state` 尚不存在、`slots.tbl` 全零且
>   每个已存在的 extent 形状正确时，重启才补建缺失文件；原文件不截断。
>   已有业务数据、未知文件、无标记的半成品以及几何不符都拒绝自动恢复。
>   启动错误包含失败系统调用的 `errno`；遇到 `EMFILE` 先核进程的软
>   `RLIMIT_NOFILE`（systemd 为 `LimitNOFILE`），再复核同一 store。
>   不得仅凭“像空仓”删除旧目录，也不得用提高 FD 上限掩盖非 FD 错误。
> - `--slab-write direct|buffered`（默认 `direct`）：slab 数据面使用 O_DIRECT；文件系统不支持时整店回退 bounded buffered，以 `wr=` 上报真值。
> - `--ram-tier on`（默认关）：arena 内对象走 RDMA 零拷贝。`--ram-write-mode writeback|writearound`（默认 `writeback`）选择 PUT 先入 RAM 后落盘，或直接盘写、读时晋升。`writeback` 默认在数据进入可读且 flush-pinned 的 RAM slot 后返回 `kOk`。达到正的 `DFKV_RAM_ACK_HIGH_WATERMARK_PCT`（默认 80）时，只拒绝尚无 owner 的新 key，返回 `kCacheFull`，不转为直接盘写。已有同 key 请求仍加入首写者的完成结果；水位设为 0 保留强制 disk-ACK 语义。需要所有 PUT 都等待持久化时设置 `DFKV_PUT_ACK_MODE=disk`。`--ram-tier-bytes` 限定总预算；超 extent 对象仍受 dedicated allocation 预算约束。显式启用 RAM 后，分配或 NUMA 模式失败会拒绝启动。
> - `--ram-tier-numa interleave|off`（默认 `interleave`）：这是当前完整 mode 集；不接受数字 node ID。arena 预触并绑策略，须核 `MemoryMax`。
> - `--ram-flush-threads <n>` / `DFKV_RAM_FLUSH_THREADS`：请求值会提高到至少 shard 数，多盘时至少 2×shard 数；实际值由 `dfkv_ram_flush_threads` 报告（默认请求=4×盘数，上限 16）。刷盘按目标盘分队列：一个 flush 批次只含一块盘的对象，同一 shard 里一块盘同时最多占用 flush 线程数减一，一块挂住的慢盘不拖住其他盘的刷盘（每 shard 只有 2 个 flush 线程时，两块盘同时挂住仍可能占满这个 shard 的全部线程）；RAM-ACK 下一块盘的脏数据最多占脏水位的 2/盘数，到了只拒绝这块盘的新 key。
> - 微调项（env only）：`DFKV_SLAB_TABLE_SYNC_MS` 控制运行期对 `slots.tbl` 的周期同步（0=关）。默认 `DFKV_SLAB_WARM_RESTART=boot` 时 100 ms，`auto` 时 5 s，`off` 时关：`off` 在非干净退出后一律清零，`auto` 只在内核和挂载都没变（页缓存还在）时保留表，两者的恢复都不依赖 100 ms 的同步；`auto` 留下 5 s 一次的慢同步，是为了在运行期间就发现表的后台回写错误（EIO），不必等到停止或重启时的同步才发现（错误只报一次，回写失败的页也可能随之丢掉，之后再读到的就是旧记录）。干净停止和 extent 改绑时表照常同步。
> - 非干净退出后的处理（env only）：`DFKV_SLAB_WARM_RESTART=off|auto|boot`，默认 `off`。
>   - `off`：非干净退出后一律清零，等同 2.35。
>   - `auto`：进程被杀（停止超时、OOM）而内核没重启时保留缓存，其余情况清零。要同时满足：`slab_state` 里记录的 boot id、挂载唯一 ID（内核 6.8 起提供）和目录 inode 都没变；`/dev/shm` 里有上次运行留下的可信令牌；文件系统是 xfs、ext4 或 tmpfs；启动时对表和各 extent 的同步没有报错。
>   - `boot`：在 `auto` 的基础上，接受上次运行没有挂载唯一 ID 的情况（内核 6.8 以前）。这时"文件系统出错下线、没重启就重新挂载"只有在旧进程退出前察觉到时才拦得住（看到过 I/O 错误，或者正在同步表，两种情况令牌都已收回）；没察觉的话，最近一次表同步之后的记录写入、以及内核最近一次回写之后的带缓冲数据写入可能丢失。缓存盘出过文件系统错误后，用 `off` 启动一次再改回来。
>   - 可信令牌是 `/dev/shm`（必须是 tmpfs）里按缓存目录命名的小文件，随重启消失，不依赖缓存盘。服务端只在能为缓存担保期间持有它：恢复完成后建立；出现元数据 I/O 错误、每次同步表期间、服务端自身崩溃（SIGSEGV、SIGABRT 等）、保留缓存的启动尚未完成时都会收回。令牌不在就清零，所以 `/dev/shm` 被清理只会多清一次缓存。
>   - 进程在同步表期间被杀也会清零（令牌此时已收回）。`auto` 下默认 5 s 同步一次，被杀正好落在同步里的机会很小；`boot` 下同步每 `DFKV_SLAB_TABLE_SYNC_MS` 做一次，写入压力大或 extent 频繁改绑时占的时间比例会上升。清零的原因可以在启动日志里看到。带缓冲写过数据的运行，在关闭阶段同步 extent 期间被杀同样清零。
>   - 服务以普通用户运行且 logind 的 `RemoveIPC=yes` 时，该用户最后一个登录会话结束会清掉 `/dev/shm` 里的令牌，之后的非干净退出都会清零；用 root 或系统用户运行不受影响。`dfkv_slab_trust_tokens_held` 可以看当前有几块盘持有令牌。
>   - 容器里 `/dev/shm` 通常是容器（或 pod）私有的，容器重建后令牌消失，`auto` 和 `boot` 都等于 `off`；内核 6.8 及以上，容器每次启动或带挂载沙箱选项的 systemd 单元挂载唯一 ID 也会变。要在容器里保留缓存，需要把宿主机的 `/dev/shm` 挂进去，并且只在内核 6.8 以前用 `boot`。
>   - 一个缓存目录同时只能被一个进程打开（目录上有排他锁）。
>   - 建议先在个别节点开 `auto` 观察 `dfkv_slab_warm_restarts_total` 和启动日志，再推广。
> - 微调项（env only）：`DFKV_SLAB_EVICT_HIGH_PCT` / `DFKV_SLAB_EVICT_LOW_PCT`
>   （默认 92 / 88；high=0 关）：越过 high 后按全局读写最近使用顺序回收 slot 到 low。
>   新键成功提交可在近期压力状态下重新触发回收；非活跃且连续 128 次无新提交观察后，历史压力失效。
>   每 tick 由 `DFKV_SLAB_EVICT_MAX_VICTIMS_PER_TICK`（默认 256，范围 1～1024）限制移除对象数，扫描上限为其 4 倍。旧 `DFKV_SLAB_EVICT_MAX_EXTENTS_PER_TICK` 不再读取，升级时须迁移配置。
>   **绝对余量（可选，默认不设）**：`DFKV_SLAB_EVICT_HEADROOM_BYTES` 设为非 0 时，high = 容量 − 余量，
>   low = high − `DFKV_SLAB_EVICT_BAND_BYTES`（默认余量的 1/4），优先于上面的百分比。下限：band 至少一个
>   extent（默认 1 GiB），余量至少两个 band（被下限抬高时打 INFO）。算出的 high 低于百分比给出的 high、
>   低于 band 或容量不够时，打 WARN 并沿用百分比（小盘上余量反而会比百分比留得多）。
>   `DFKV_SLAB_EVICT_HIGH_PCT=0`（关水位）时设了余量会重新打开水位回收，同样打 WARN。生效值记在配置快照
>   （`DFKV_SLAB_EVICT_HEADROOM_BYTES`、`DFKV_SLAB_EVICT_BAND_BYTES`，0 表示用百分比），每盘的 high/low
>   见 `dfkv_slab_disk_watermark_high_bytes` / `_low_bytes`。
>   为什么：92/88 让 8–12% 的容量常年空着（每 TiB 约 80–120 GiB），而余量只需盖住两次后台回收之间的写入；
>   单尺寸档的单元级测量里，余量小到水位回收不触发时，淘汰全部在分配路径完成，PUT p99 反而从 2.18 ms
>   降到 0.43 ms（水位回收每 tick 持锁 0.3–0.5 ms）。多尺寸档竞争下的效果还没测到。
>   建议：1 TiB 以上的盘可试 `DFKV_SLAB_EVICT_HEADROOM_BYTES=8589934592`（8 GiB）、
>   `DFKV_SLAB_EVICT_BAND_BYTES=2147483648`（2 GiB），即 high 只比容量低 8 GiB。先在个别节点上线，
>   对比前后的 `dfkv_slab_watermark_evictions_total`、`dfkv_slab_disk_watermark_high_observations_total`、
>   kCacheFull 与 `put_busy`、PUT p99。**默认值只在有了真实部署的这组观测之后才会改。**
> - `--slab-reclaim-ms <n>` / `--ram-reclaim-ms <n>`（默认 50 / 10，`0`=关）：后台预回收和类再平衡。类再平衡抢占（`StealFrom`）与 Put 的条带下限/兜底抢占（`StealExtentFor`）优先选空 extent，其次选驻留对象平均最久未访问的 extent（decayed read heat、useful bytes 仅作平手裁决），跳过 pinned extent；这是 extent 级取舍，混在大量旧对象中的少量新对象仍可能随整块淘汰；冷抢占（`StealColdExtentFor`）保持原有入驻年龄门槛与 heat/useful bytes 排序；后台类再平衡按写入字节（而非对象个数）判断冷 donor class；通常保留默认。
> - `--slab-granularity <bytes>`（默认 1 MiB）：最小 slot 量子。现有 `slots.tbl` 的 format/geometry 不匹配会 fail closed；修改必须换空目录，服务不会原地冷重建或忽略旧数据。
> - **`slots.tbl` 占用与预留**：每盘 `盘容量 / granularity × 64` 字节，在 `--cap` **之外**（默认 1 MiB 量子时约为容量的 1/16384）。启动时用 `fallocate` 整体预留，已分配齐的表不再处理；文件大小和 record 格式不变，旧版本仍可打开。**新建目录**：直接分配，文件系统放不下时该盘启动失败（`cannot reserve N bytes for slots.tbl ... No space left on device`）。**已有目录**（含旧版本写出的稀疏表）：始终照常启动；升级后首次启动补齐（日志 `reserved slots.tbl, N bytes (M were allocated)`），文件系统放不下时不拒绝启动，保持原先的惰性分配并在每次启动打一行 WARN `no room to reserve slots.tbl, X of N bytes are not allocated`——该盘仍有“文件系统写满时 store 终态故障”的风险，腾出 X 字节后重启即补齐。direct 模式下 extent 也在启动时整体分配；表与 extent 都预留成功后，在预留后原地覆盖写的文件系统（xfs、ext4）上 store 不会再因文件系统写满而失败，其他文件系统（写时复制、tmpfs 等）不作此保证；buffered 模式（`--slab-write buffered` 或文件系统不支持 O_DIRECT）的 extent 仍按写入惰性分配，文件系统写满时对应 PUT 失败（store 不因此进入终态故障），容量规划仍须保证 `--cap` 加 `slots.tbl` 不超过文件系统可用空间。不支持 `fallocate` 的文件系统打一行 WARN 并保持原先的惰性分配。
> - `--put-inflight-limit <n>`（默认 0=关）：在 RAM 接纳或直接盘写之前原子限制前台 PUT 并发；超限返回 `kCacheFull`，不接纳数据、不进入 peer cooldown。TCP 与 RDMA direct PUT 共用此门；后台 flusher 不受限制。已接纳写入和同 key 重复请求仍等待原写入的终态，不以超时将另一份 payload 送入盘写路径。该参数限制接纳量，不是磁盘 I/O 超时保证。
> - `--tcp-max-conns <n>` / `DFKV_TCP_MAX_CONNS`（默认 512，硬上限 4096）限制
>   TCP handler/FD；`--tcp-io-timeout-s <n>` / `DFKV_TCP_IO_TIMEOUT_S`（默认
>   60s，硬上限 3600s）回收 silent/半帧连接。达到上限的新连接会被立即拒绝，
>   pooled TCP 客户端重连；监控 §7 的 reject counter。
> - `DFKV_TCP_FIRST_REQ_MS`（默认 30000 ms，0=关，硬上限 3600000）：首帧
>   absolute deadline，从 accept 起第一段完整 request frame 必须到齐。逐字节
>   滴喂的慢连接不再永久占用 handler 槽位（SO_RCVTIMEO 只约束单次 syscall）；
>   每次读的 per-syscall 预算取 min(base timeout, deadline 剩余）。resolved
>   值由 `dfkv_tcp_first_req_ms` gauge 上报（见 METRICS.md §3.1）。
> - RAM 热层 arena 在启动期预触页：**4K 页时实测 192 GiB 约 39–47 秒（每 16 GiB 约 4 秒）**，拿到透明大页时只要几秒（见下）。大页是尽力而为的，所以 systemd `TimeoutStartSec`（默认 90s）仍按 4K 页的耗时随 arena 调大，并让就绪探测等待 metrics 端口。arena 预触页和各盘的打开（恢复 `slots.tbl`）同时进行，启动耗时约等于两者中较慢的一个，而不是两者之和；日志里 `ram tier arena pre-faulted` 和各盘的 `slab <目录>: opened in ...` 因此先后不定。某块盘打开失败时进程照旧启动失败，但会等 arena 预触页结束后才退出。恢复期间每块盘临时占用的内存（230 万对象时最多约 250 MiB）现在与 arena 预触页同时存在，内存按 arena 加这部分留余量。
> - arena 在预触页时用 `mlock2(MLOCK_ONFAULT)` 锁在内存里，不会被换出（需要足够的 memlock 上限，样例单元里是 `LimitMEMLOCK=infinity`）。锁不上只打一条 WARN，服务照常运行，这时开了 swap 的主机在内存紧张时可能把 arena 换出。`DFKV_RAM_TIER_LOCK=off` 可以关闭，`dfkv_ram_arena_locked` 指标反映实际有没有锁上。以前这块内存是因为整块注册到各 rail 而被 pin 住的；只走 TCP 的节点以前没有这层保护，现在也会锁。
> - slab 盘在打开时不标记为使用中：打开只读状态、重建索引，`slab_state` 保持上次停止时的内容，不写信任令牌，不启动后台线程，也不接受任何读写。打开时仍可能做三件每次打开都可以重复的事：给 extent 补分配缺失的块、为 `slots.tbl` 预留空间、清掉校验不过的记录。全部盘打开完成、RAM 层就绪之后，在开始监听之前一起“激活”：写入并同步 dirty 状态和本次运行的身份、发放信任令牌，然后才启动同步和回收线程、放行操作，日志里有一行 `disk stores activated: N in X ms; an exit without a clean stop is an unclean exit from here on`。在这一行之前进程被杀（SIGKILL、OOM）、启动被信号取消或启动失败（某块盘打不开、RAM 层初始化失败），每块盘都保持上次停止时的样子，下次启动是干净启动、缓存保留，与 `DFKV_SLAB_WARM_RESTART` 的取值无关；这一行之后非干净退出的处理和以前一样。上次没有干净停止的盘（打开时发现是 dirty）照旧在打开时清零或保留，并当场激活。
> - 启动期间收到 SIGTERM 或 SIGINT 会取消启动，而不是等启动跑完再停：arena 预触页在信号后一个 2 MiB 步长内放弃；各盘的打开不中断（秒级），已经打开的盘只同步、保持原样（这时它们还没有被标记为使用中，不写状态文件），下次启动是干净启动。取消点有三处：各盘打开之后（此时还没有监听 TCP）、RDMA 监听启动之后、健康检查端口启动和向 MDS 注册之前；取消的启动不会注册。注册循环启动之后收到信号就是普通的停止。日志最后一行是 `dfkv_server start cancelled: not registered with the MDS, every opened disk closed clean, N s after the process started; exiting`，退出码 0（有盘打开失败或没能写 clean 时是 WARN 行 `... but a disk failed to open or is not clean ...` 和退出码 1）。arena 预触页开始时打一行 `ram tier arena: faulting in N MiB (threads=T)`，4K 页的大 arena 在这之后会有几十秒没有新日志。
> - 停止顺序（收到 SIGTERM 之后）：先给正在等刷盘的 PUT（见 `DFKV_RAM_PUT_WAIT_MS`）回 `kCacheFull`，停请求不再等它们的刷盘 → 停 RDMA 和 TCP 请求 → RAM 层积压刷盘（时限 `DFKV_RAM_SHUTDOWN_DRAIN_TIMEOUT_S`，默认 30 秒，超时丢弃剩余）→ 各盘同时同步并写 clean → 进程直接退出（内存交给内核回收，不再逐个对象释放）。每一步在日志里有一行 `shutdown: ...` 和耗时（INFO 级别）；最后一行是 `dfkv_server shut down cleanly in N s; exiting`，这一行之后进程即使被 `TimeoutStopSec` 杀掉，下次启动也是干净启动，缓存保留。2.36 及更早版本 clean 标记写在释放内存之后，那段时间被杀会清空全部缓存。
> - 启动时每块 slab 盘打一行 `slab <目录>: opened in N ms, K objects restored (extents A ms, allocation skipped for S of E; table scan B ms; restore C ms)`（INFO 级别）：`extents` 是打开全部 extent 文件的时间，`table scan` 是读并校验 `slots.tbl` 的时间，`restore` 是把恢复出的对象装入内存索引的时间。`allocation skipped for S of E` 只在 direct 模式下出现（buffered 模式启动时不对 extent 调用 `fallocate`）：上次干净停止且 extent 文件已经分配满时不再对它调用 `fallocate`，`S` 是跳过的 extent 数；非干净退出后的启动、新建的或有空洞的 extent 照旧调用，这时 `S` 小于 `E`（非干净退出后为 0）。启动变慢时先看这一行是哪一段变长。关闭一块盘时如果还有数据写入没有结束，这块盘不写 clean 标记，打一条 WARN，下次启动按非干净退出处理；正常停止顺序（见上一条）先停请求和 RAM 层刷盘，不会走到这里。
> - arena 默认申请透明大页。4K 页下内核每秒只能给出约 4 GiB，加线程没有用，退出时归还也慢；2 MiB 页下 192 GiB 的 arena 实测 3.3 秒常驻、约 1 秒归还（4K 页是 39–47 秒和 13–21 秒），从 RAM 层读 1 MiB 对象的吞吐高 30–40%。启动日志 `ram tier arena pre-faulted: ... huge pages=N MiB` 和指标 `dfkv_ram_arena_hugepage_bytes` 给出实际由大页承载的量：系统设置是 `never` 时为 0；空闲内存碎片化时会小于 arena，而且内核在分配时做内存规整，预触页可能比 4K 页还慢（没有实测过这种主机）。遇到这种情况可以设 `DFKV_RAM_TIER_HUGEPAGES=off`：不再申请，页大小回到由系统设置决定（`madvise` 下是 4K 页；`always` 下 arena 本来就在大页上，页大小关不掉，但不申请之后内核在缺页时不再同步做内存规整，指标因为没有测量而读 0）。
> GPU 节点推荐：保留默认 slab/direct，按节点突发画像启用 RAM tier 并设置总预算。
```bash
systemctl daemon-reload && systemctl enable --now dfkv
journalctl -u dfkv -n 10 --no-pager
# 应见 "PORT 28000" + "rdma topology startup: configured=8 initialized=8 ACTIVE=<n> inactive=<names|(none)>"
#      + "dfkv_server MDS registration loop started group=default id=n57 advertise=192.168.1.57:28001"
# 首次成功后另见 "dfkv_server registered with MDS group=default id=n57"
```
> `--mds-registration-timeout-ms` / `DFKV_MDS_REGISTRATION_TIMEOUT_MS` 是首次
> MDS 注册硬截止时间（默认 60000 ms，合法范围 1000–600000 ms，flag 优先）。
> 非整数、越界值均以配置错误退出 2；截止前仍未注册则停止所有 listener 并退出
> 1，由 `Restart=on-failure` 重试，节点不会无限保持 active-but-unregistered。
> 心跳间隔 `DFKV_MDS_HEARTBEAT_MS`（默认 10000，范围 1000–10000，超出按上限）。约束是心跳间隔加注册器一次请求的 I/O 超时（现为 7 s）小于 MDS 租约（30 s）：一次心跳迟到一个超时也赶得上续约。上限 10 s 留出余量，I/O 超时提高到 11 s 时也满足（编译期检查）。
> `GET /healthz` 反映当前 store/RAM terminal health；`GET /readyz` 还要求本地
> startup 和首次 MDS 注册完成。首次成功后 readiness latch 不因瞬时 heartbeat
> 失败清零，已注册节点继续运行并在 MDS 恢复后续租；必须同时监控
> [METRICS.md](METRICS.md) 的 registrar heartbeat 指标。
> RDMA startup 日志
> `rdma topology startup: configured=N initialized=N ACTIVE=M inactive=...`
> 分别给出显式 configured、完成 anchor/启动 MR 的固定 initialized 集合、以及
> 启动瞬间 ACTIVE 快照。显式部署必须 `configured == initialized`；missing/open/
> port/GID/anchor/MR 任一失败仍拒绝启动。运行期不可用 rail 不会被删除或重排。
> 当前总数和健康数看 `dfkv_server_rdma_rails_configured`、
> `dfkv_rdma_rails_initialized`、`dfkv_server_rdma_rails_active` 和
> `dfkv_server_ib_device_healthy{device}`，不能把启动 `ACTIVE` 当现状。这些
> family 仅在 RDMA listener 存在时输出；TCP-only 进程缺失是预期行为。
>
> **partial health**：至少一条 initialized rail 健康即允许 placement。9→8 等
> 部分掉轨保持 ring eligible，并报告 `PARTIAL`；最后一条健康轨丢失时立即
> `DEGRADED`/退环。0→1 恢复保持
> `DFKV_RDMA_HEALTH_RECOVERY_SAMPLES`（默认 3）连续成功采样门，通过后以
> `PARTIAL` 或 `ACTIVE` 原进程回环，无需重启。该门只防恢复抖动，不延迟从非零
> 到零的摘除。
>
> **disk store 终态故障**：slab 元数据写入/落盘失败（或 io_uring 丢失完成事件）
> 后，该 store 在本进程内不再读写，只有重启能恢复。故障时 store 打一行 ERROR
> `slab <dir>: disk store failed for good: cannot <操作>: errno=...`。该状态是只置不清的
> latch，不会抖动。
>
> - **多盘节点降级运行**（只要还剩 `DFKV_DEGRADED_MIN_DISKS` 块健康盘，默认 1）：坏盘
>   的 key 改由其余健康盘承接（每块坏盘的 key 均匀分到各健康盘；健康盘自己的 key 不
>   动），在那里先是未命中（GET 返回 `kNotFound`，不是 I/O 错误，client 不会因此回避
>   整个节点），重新写入后照常命中。节点留在 placement 里，`/healthz` 仍为 200，
>   `dfkv_server_ring_eligible=1`；`dfkv_disks_failed`、`dfkv_disk_failed{disk}`、
>   `dfkv_server_degraded=1` 和 INFO 末尾的 `store=degraded` 表明降级，节点打一行
>   ERROR `N of M disk stores have failed ...`，告警 `DfkvServerDegraded`。节点的 key 份额
>   不变而容量少了坏盘那一份，所以它的淘汰会比其他节点早。
>
>   承接之前，节点先把故障记在每块健康盘的缓存目录旁边（`<缓存目录>.dfkv-failed-disks`，
>   例如 `/mnt/disk2/dfkv.dfkv-failed-disks`，内容是故障盘目录的规范路径）。记录只写在
>   上一级目录与缓存目录同属一个文件系统的地方，这样它和这块盘上的数据同生同灭；所以
>   **每个缓存目录都要是所在盘文件系统里的子目录**（如 `/mnt/diskN/dfkv`），不能是挂载点
>   本身，也不能是容器卷的根（上一级目录不在这块盘上，容器重启就没了）。一处都写不了
>   时不承接，节点按健康盘不足处理，并打一行 ERROR。服务账号要能在上一级目录里建文件。
>
>   下次启动看到记录，先把记录里的盘整盘清空（由各引擎自己重置，不逐 key 删除），再从
>   其余各盘删掉不属于本盘的 key（上一轮代坏盘承接的），所以重启后不会冒出坏盘里的旧值，
>   也不会冒出承接期间写下、之后已被 REMOVE 或覆盖的值。若某份记录存在却读不全，或写着
>   本节点没有的目录（例如目录换了位置），无法知道当时哪块盘坏了，就清空**所有**盘并打
>   ERROR（缓存语义，不拒绝启动）。删掉的数量见启动日志 `disk failure records found ...`
>   和 `dfkv_disks_open_dropped_keys{reason="failed_disk"|"not_home"|"unusable_record"}`。
>   某块盘上不属于它的 key 删不掉时，这块盘本轮不再承接坏盘的 key
>   （`dfkv_disk_takeover_excluded{disk}=1`，WARN），记录保留到下次启动重试；有盘故障时这样
>   的盘不计入 `DFKV_DEGRADED_MIN_DISKS`，健康盘全都不承接时节点按健康盘不足退出 placement。
>   删掉的 key 先同步落盘（file 引擎做一次 syncfs）；同步失败时记录同样保留。记录在之后
>   一次没有盘故障、各盘都干净关闭的运行停止时删除；记录路径上若是目录等非普通文件，
>   节点不删它（WARN），每次启动都会因此清空所有盘，需要运维移走。
> - **退出 placement**：健康盘少于 `DFKV_DEGRADED_MIN_DISKS`（单盘节点坏一块盘即是）时，
>   PUT/GET 返回 `kIOError`，Exist/Retain 按 miss 回答，前置 RAM tier 仍命中的条目照常
>   返回。节点在下一次心跳（≤10 s）自报不可 placement：MDS 不再把它放进 client 的成员
>   列表，其 key 份额迁到其他节点；它保持注册，`dfkvctl ring` 显示 `DEGRADED`、INFO
>   末尾带 `store=failed`，`dfkv_server_ring_eligible=0`、`/healthz` 503。TCP-only 节点
>   同样适用（健康时仍不发 HLT1，故障后发一个不含设备的 HLT1）。坏盘的 key 此时不
>   改道，留在坏盘上（返回 `kIOError`，RAM tier 也不再确认它们）。把
>   `DFKV_DEGRADED_MIN_DISKS` 设为盘数（或更大）即恢复“任一盘故障整节点退出”的旧行为。
>
> 进程不自行退出：重启会清空该盘（slab 自己重置；其他引擎按上面的故障记录清空）并
> 重新启用它，若盘确实坏了会再次故障；
> 因此由运维决定重启（故障已排除）、换盘（见 §2 末尾的容量说明：目录列表与 `--cap`
> 不变）或用 §4c 替换节点。启动时打不开的盘仍使整个节点启动失败，不降级。
> slot table（`slots.tbl`）在新建目录时整体预留（`fallocate`），预留成功后写 record
> 不再需要新 block；只有**新建目录**会因文件系统放不下它而启动失败。已有目录始终
> 启动：表没分配齐且文件系统没有空间时，每次启动打一行 WARN
> `no room to reserve slots.tbl, X of N bytes are not allocated`，表示该盘仍可能在
> 文件系统写满时故障。因此 `store=failed` 的节点若是因文件系统写满而故障（故障行里
> `errno=28`），重启前先腾出空间：否则重启要么在重置表时失败，要么带着上述 WARN
> 启动后再次故障。
> 需要 MDS ≥ v2.15.0 才会按 HLT1 过滤；更老的 MDS 忽略该字段，行为与此前相同。
> RAM tier 丢弃无法 flush 的条目只使 `/healthz` 变 503（最后一次丢弃至少 30 秒之后、且
> 连续 64 次 flush 成功才恢复），
> 节点仍可服务，不退出 placement。盘 store 终态故障后，坏盘的 key 改写到其余健康盘，
> RAM 照常确认；故障瞬间已确认、尚未落盘的值在重试时写到承接它的健康盘。没有健康盘
> 时，PUT 不再由 RAM 确认，直接得到 `kIOError`（与没有 RAM tier 时相同），故障瞬间已
> 确认、尚未落盘的值无法挽回，立即丢弃并计入
> `dfkv_ram_post_ack_flush_failures_total`（告警 `DfkvRamAcknowledgedDataLost`）。
> 盘侧瞬时写失败（一次 EIO、store 在高负载下暂时放不下）时，已确认的值留在 RAM 里按
> 50 ms→3.2 s 的间隔重试约 4 秒，仍失败才丢弃并计数；文件系统写满这类持续数分钟的
> 故障超出这个窗口，值会被丢弃（计数、告警、WARN），腾出空间后节点自行恢复健康。
> 重试期间这些值仍 pin 在 RAM 里并计入 dirty：盘一直不收写时，最多占到 RAM tier 的
> ACK 水位（`DFKV_RAM_ACK_HIGH_WATERMARK_PCT`，默认 80%），超过后 client 的**新 key**
> 得到 `kCacheFull`（不触发 peer cooldown），读和已驻留的 key 照常。**该水位是整个
> tier 一个数，不分盘**：多盘节点上只要一块盘不收写就可能把它占满，此时该节点**所有盘**
> 的新 key 都得到 `kCacheFull`，直到那块盘的值被丢弃或写入为止。同一 key 的重复 PUT
> 在其值已由 RAM 确认、仍在等落盘时立即得到同样的确认，不等待落盘。
> 盘**健康但暂时没有槽**（slab 盘回 `kCacheFull`：该 size class 的槽全被在途读写 pin
> 住或被留存保护）不算写失败：不计重试次数、不影响 `dfkv_ram_healthy`。已确认的值留在
> RAM 里按同样的间隔（最长 3.2 s 一次）继续试，最长 60 秒，期间可读、计入 dirty；
> 60 秒内仍没有槽才丢弃，计入 `dfkv_ram_flush_no_room_dropped_total`（相当于一次淘汰，
> 不计入 `dfkv_ram_post_ack_flush_failures_total`，不触发 `DfkvRamAcknowledgedDataLost`）。
> 停止时排空（drain）期间不再等槽：这样的值和写失败的值一样不加间隔再试
> `flush_retries`（默认 5）次，仍没有槽就丢弃，同样只计入
> `dfkv_ram_flush_no_room_dropped_total`。挂在这类值上的 RETAIN SET 保留到值真正写入
> 时才生效，值最终被丢弃或删除时一并作废。
> 等落盘结果的 PUT（disk-ACK）则立即得到 `kCacheFull`，与没有 RAM tier 时盘的回答相同。
> **行为变化**：`dfkv_ram_healthy` 以及由它决定的 `/healthz`、`/readyz` 不再是“一旦为 0
> 直到重启”，依赖 503 保持不变的探针应改看 `dfkv_ram_flush_dropped_total` /
> `dfkv_ram_post_ack_flush_failures_total`（两者保持单调）。数据面线程卡死而进程存活的情况不在此检测范围内，依赖
> `DfkvServerNotReady`（抓取失败）与 client 侧错误告警发现。
>
> **peer-aware tiers**：server `--rdma-dev` 和 client `DFKV_RDMA_DEV` 都固定
> 本机设备索引。client 的 `DFKV_RDMA_RAIL_TIERS` 用 `a|b;c` 表示优先级：
> `|` 是同 tier，`;` 从 Tier 0 向低优先级分隔；空 tier/name、重复 rail 或任何
> 不在 `DFKV_RDMA_DEV` 的名称都使 client 初始化失败。未配置 tiers 时全部 local
> devices 是一个 homogeneous tier。MDS `kListTopology` 返回 Members 基础记录的
> 地址/placement binding，并复用 `HLT1` 扩展承载 eligibility 与
> `{device,port_state,phys_state,query_ok}`；不另建 rail 协议。client 选择稳定的
> `local enabled ∩ peer healthy ∩ highest available tier`。NUMA、credits、
> latency、quarantine/cooldown 只在该 tier 内排序/等待；**健康 Tier-0 credit
> 紧张必须等待，拥塞绝不溢出 Tier 1**。只有 peer health 使高 tier 完全不可用，
> 才会使用下一 tier。
>
> **rail 同名配对**：client 在 bootstrap 里报本端设备名，server 在同名设备上建 QP，没有“本端 X 连对端 Y”的交叉配对；候选集是 `DFKV_RDMA_DEV`，列表外的对端 rail 不用。对端不服务某个设备（server 在回复前关掉 bootstrap）时，第一次照旧按“忙”处理（停一下换 rail 重试）；若同一操作随即在另一条 rail 上连上了这个 peer，client 在原 rail 上再拨一次确认：仍在回复前被关掉，才认定该设备被拒，把 (peer, rail) 记为一次 endpoint 失败、进入冷却（之后有恢复探测），计数 `dfkv_rdma_client_rail_refusals_total`，后续操作不再先撞它。确认拨号的连接和每次读最多等 300 ms，超时不算确认、也不计任何 rail。握手满载（`DFKV_RDMA_MAX_PENDING_HANDSHAKES`）或 server 停止时的关闭在确认拨号时通常已不再出现，不会被当成拒绝；若对端恰在确认拨号时仍满载或处于 receive-pool 压力，也可能被计为拒绝，代价有界（基础冷却 `DFKV_RDMA_REMOTE_RAIL_COOLDOWN_MS`，默认 2 s，之后有恢复探测）。对端在本端可用（端口正常、未本地隔离）的 rail 只剩这一条时不冷却，同一对端的两次确认不会并发判定。静态成员（没有 MDS 身份）按地址记这类拒绝，在任一 rail 上新建连接成功即清掉该 rail 的失败计数；普通 endpoint 失败对静态成员仍不冷却。
>
> **本端端口状态**：client 每 `DFKV_RDMA_PORT_CHECK_MS`（默认 1000，0 关闭，上限 60000）查一次各 rail 的端口状态（每条 rail 一个常开的设备句柄，`ibv_query_port`）。端口离开 ACTIVE 的 rail 退出选择，其空闲连接回收，日志 WARN `port not ACTIVE`；回到 ACTIVE 后重新参与选择（INFO）。查询失败按状态未知处理、保持原样。启动时就不是 ACTIVE 的 rail 不在候选里（发现阶段只取 ACTIVE），运行中恢复也不会加入，需重启 client。指标 `dfkv_rdma_client_rails_port_down`、`dfkv_rdma_client_port_down_events_total`。
>
> **server 端口状态**：server 用同一个 `DFKV_RDMA_PORT_CHECK_MS` 在单独的线程里查各 rail 的端口（server 上默认 0，即关闭，行为与以前相同；建议设 1000，下面说的都是打开后的变化）（不占握手或服务线程，查询时不持这些线程要的锁）。端口不是 ACTIVE 的 rail：新连接在握手时直接关掉、不回答（`dfkv_rdma_inactive_rail_refusals_total`），client 当作对端忙暂停约 20 ms 再试一次，第二次仍被关就冷却 server 的这条 rail、改走别的 rail，不再在建好却走不通的 QP 上等 RC 重试耗尽；只有这一条 rail 的 client 立刻得到失败。端口连续两次查询都不是 ACTIVE 时（只有一次时只拒绝新连接，链路在一个查询周期内重新训练好的话已有连接不受影响），这条 rail 上已有的连接立即结束（只动这条 rail，`dfkv_rdma_rail_down_connections_ended_total`），释放它们的 pin 和接收槽；开了 `DFKV_RDMA_SESSION_FD` 的 client 复用前就能发现，否则下次使用时照旧等一次 RC 超时。状态变化（每条 rail 每秒至多一次；一秒内被压下的变化，在一秒过后的第一次查询时若仍成立再补发）会让 server 立刻向 MDS 发一次心跳，健康信息随之更新，client 下一次轮询（约 3 s）就看到，而不是最多等一个心跳间隔（10 s）。不指定设备的连接分到第一条不是 DOWN 的 rail。查询失败按未知处理，保持上次看到的状态；端口查询一直失败的设备（状态 -1）永远不会被确认为 DOWN，它上面的连接照旧靠自身的失败结束。`dfkv_rdma_rail_port_active{dev}` 是各 rail 最近一次看到的状态。“端口 ACTIVE 但路径不通”这里检测不到。
>
> GPU/CPU 异构示例（名称仅为示例，生产须用两端实际同 fabric 名）：
> ```bash
> # GPU host: two preferred rails plus one lower-bandwidth fallback
> export DFKV_RDMA_DEV=mlx5_0,mlx5_1,mlx5_2
> export DFKV_RDMA_RAIL_TIERS='mlx5_0|mlx5_1;mlx5_2'
>
> # CPU host: one preferred rail plus the shared fallback
> export DFKV_RDMA_DEV=mlx5_0,mlx5_2
> export DFKV_RDMA_RAIL_TIERS='mlx5_0;mlx5_2'
> ```
> GPU↔GPU 可在 `mlx5_0|mlx5_1` 内选择；GPU↔CPU 的最高共同健康 tier 是
> `mlx5_0`，仅它健康性消失后才考虑 `mlx5_2`。设备名不匹配不是“异构映射”，
> 而是无交集。
>
> **HLT1 / mixed version**：`MembersTopologyEpoch` 以稳定 member-id/device-name
> 顺序 canonicalize 地址/placement binding、eligibility 和每轨 health 字段；
> client 用 placement epoch 独立去重 Ketama ring，同时始终把新的 topology
> generation 送给 transport。旧 MDS 不识别 `kListTopology`：modern poll 会失败并
> 保留 last-good ring，不会静默降回 `kListMembers`；因此 rollout 先升级 MDS。
> 当前 MDS 中的旧 member 没有 HLT1，topology 标记 incomplete。未配置 tiers 的
> homogeneous client 可继续服务这类 member；配置 tiers 时返回
> `Status::kNoCompatibleRail`。静态 member 列表也没有 HLT1，不能与 configured
> tiers 搭配。旧 client 对新 MDS 仍可使用既有 placement API，但没有 peer-aware
> tier 能力。
>
> **generation fence 与错误语义**：每个 logical operation 在开始时捕获一个
> peer generation，两次物理尝试都使用它。Conn 记录 peer generation；旧代 pool
> entry 不再复用/回池，只由既有 single-owner lifecycle teardown，防止 topology
> 变化后跨代 QP 复活。`kNoCompatibleRail` 是 client-local、从不编码上 wire，
> 不计 served response、IO error 或 peer cooldown。
>
> 物理尝试总数仍固定至多 2。已完成的 GET/batch/SG item/window 保留 byte-exact
> 结果，第二次只提交未完成工作。PUT 只有在确认 request 尚未 post 时才可换轨；
> post 后 failure 的远端 commit 状态不明确，必须原样失败，**不得跨轨 replay**。
> 唯一例外是 stale pooled QP：连接从池里取出后没有任何 completion 成功（连对端
> 网卡对请求的确认都没有）就失败，请求几乎可以肯定没有送达（对端已回收该连接或
> 已重启），这样的 PUT 在 fresh connection 上再发一次，与读相同。若它其实送达了
> （只是回程的确认全部丢失，或请求的 completion 恰在窗口期限到时才到），再发的
> 是同一 key、同一字节，server 按已有的那份应答；送达的第一份仍可能晚于第二份
> 才被处理，这与此前 client 放弃的 PUT 是同一种情况。直接写入调用方内存的整对象
> GET 同样只在这种情况下重试一次。这类 PUT、批量 PUT 和整对象 GET 的最坏耗时
> 加倍：第一个窗口要先失败（等到重传耗尽或窗口期限），再跑一个完整的第二窗口。
> 这不承诺 exactly-once 或 concurrent-writer isolation。
>
> client-local rail quarantine 仍由本地 Open/QP/MR/post/CQ 和 local WC 证据
> 驱动；peer/bootstrap/protocol、remote/RNR/timeout、admission、
> `kNoCompatibleRail` 均不增加 local rail error。cooldown 到期仍只准入一个真实
> recovery probe。
>
> **receive-pool 预算**：`slot=align4K(4096 + block_class)`。
> 旧 peer 的 resident lease 是 `2 × depth_class × slot`；动态 pull peer
> 只保留 `depth_class × slot` 的 receive slots，直接 GET 使用 minimum class。
> 在飞 PUT/GET 租约也计入 `DFKV_RDMA_RECV_SEGMENT_SIZE` hard budget。
> 上线同时观察 committed/used/真实 RSS 与动态租约归还；不能把操作结束后的
> used_bytes 当作整个运行过程的峰值或 pinned memory。
> 仅当初始 chunk 容得下最大合法暂存对象，且 hard budget 除初始
> chunk 外还能为**每条已配置 rail** 保留至少一个连接 chunk、
> 另有一个暂存余量时，才将初始 chunk 专供动态 GET / leased PUT。
> 连接驻留 lease 不占据它，预留量**已包含在** hard budget 内；
> chunk 小于最大对象、总预算小或多 rail 几何不足时保持共享模式。
> `dfkv_rdma_recv_segment_free_bytes` 含暂存专用空闲，不能拿它判断新连接
> 能否准入；另看 `connection_free_bytes`、连续空闲范围和 staging
> reserve 空闲。预算全部由活动连接/暂存占用时仍会如实拒绝，不绕过上限。

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
重复 hash 或多余字段都会令节点 fail closed（启动时）。运行中改了文件后给 `dfkv_server` 发 `SIGHUP` 即重读，新限额对重读之后开始的写生效（正在进行的写按原限额）；`DFKV_TENANT_DEFAULT_QUOTA_BYTES` 是环境变量，不随 SIGHUP 重读；重读时文件不合法则保留原限额并打 WARN，`dfkv_tenant_quota_reloads_total` / `dfkv_tenant_quota_reload_failures_total` 计数。hash 算法由
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

### 3a-1. slab 槽粒度：小对象占比高的负载

`DFKV_SLAB_GRANULARITY`（默认 1 MiB）是 slab 最小槽。每个对象至少占一个槽，
所以一页里有多个远小于 1 MiB 的对象时，盘上大部分空间存的是取整余量。典型是
带逻辑锚点池和多个侧池的稀疏注意力类模型（如 DeepSeek V4 / DSA 类）：每页写
一个 1 字节的锚点标记和若干约 100 KB 的侧池对象。

**实测**（单元级测量：每页 1 个 1 字节标记 + 4 个 100 KB 对象，每 30 页一个
6 MB 对象；128 MiB 小存储、8 MiB extent、8 个写线程，写入 1.5 倍容量）：

| 粒度 | 盘上实际数据 / 容量 | 留存对象数 |
|---|---|---|
| 1 MiB（默认） | 15.9% | 114 |
| 128 KiB | 65.1% | 646 |

即同样的盘约多存 4 倍数据。64 KiB 在这个小存储里是 42.6%（三个尺寸档争用
16 个 extent 的放置效应，不代表大盘），不比 128 KiB 好。

**建议**

- 这类负载用 `DFKV_SLAB_GRANULARITY=131072`（128 KiB）。
- 64 KiB 只在内存充裕时考虑（见下面的代价），本次测量没有显示它比 128 KiB
  多存数据。
- 大对象为主（每对象 1 MiB 以上）的负载保持默认 1 MiB。

**代价**（除注明"实测"外均为按格式和数据结构计算的估算）

- **每盘一次冷启动，需人工清目录。** 粒度写在 `slab_meta` 里；已有目录用不同
  粒度启动会被拒绝（`existing slab layout differs from requested geometry`），
  不会自动重建。改粒度 = 停服务、清空该盘缓存目录、再启动，缓存从空开始。
- **slots.tbl 变大**（计算）：大小 = 容量 / 粒度 × 64 B，按整盘预分配，每次
  启动都要扫描一遍。

  | 粒度 | 每 TiB |
  |---|---|
  | 1 MiB | 64 MiB |
  | 128 KiB | 512 MiB |
  | 64 KiB | 1 GiB |

  64 KiB 时启动扫描估计每 TiB 多约 0.3～0.6 秒；各盘目前按顺序打开，会累加。
- **内存随对象数增长**（计算）：每个留存对象约 200 B（索引与元数据）。按上面的
  对象组合，满盘时 128 KiB 约 1 GB / TiB，64 KiB 约 1.5～2 GB / TiB；默认 1 MiB
  约 0.25 GB / TiB。多出来的内存对应多存的对象，上线前按节点总盘量核对内存预算。
- **全量测试**：在 128 KiB 默认粒度下全量单测通过；slab 自身的测试另外显式覆盖
  4 KiB 粒度。

**水位回收的移除上限**

后台水位回收每 tick（`DFKV_SLAB_RECLAIM_MS`，默认 50 ms）最多移除
`DFKV_SLAB_EVICT_MAX_VICTIMS_PER_TICK`（默认 256）个对象。粒度变小后同样的上限
释放的字节变少（64 KiB 槽时 256 个只有 16 MiB / tick）。若
`dfkv_slab_watermark_evictions_total` 跟不上写入（`dfkv_slab_disk_watermark_active`
长时间为 1、`dfkv_slab_disk_watermark_high_observations_total` 持续增长），把它
调大（上限 1024）。实测（单尺寸档）：上限过小时更多淘汰转到分配路径完成，PUT
延迟没有变化，所以这是吞吐调节项，不是延迟调节项。

### 3b. dfkv_server flag / env 参考矩阵（按类别）

以下均为 v2.0.0（cbe53fa）时刻从 `dfkv_server --help`、源码 `Env*`/`args.get` 解析点与常量实测
整理的**权威服务端配置参考**。flag 优先于同名 env（`--rdma-dev` 直传 RDMA server，其余行为
flag 为 env facade）；未列 flag 的全部 env 均从源码排查就不误报。**客户端
侧（DFKV_RDMA_DEV/DFKV_LIB/DFKV_FANOUT_THREADS 等在同一 host 上的 inference rank 进程）仍然
按 CONNECTORS.md §1 的 vLLM/SGLang/LMCache 通用表设置**，本表只列 **dfkv_server / dfkv_mds**
进程读到的组。

**值的格式（dfkv_server / dfkv_mds）。** 数值一律是十进制非负整数，不带单位、符号、
小数或指数（字节数写全，`64G`、`30s`、`1e3` 都不是合法值）；开关写 `1` 或 `0`；
`--log` / `DFKV_LOG` 的级别大小写均可。未设置等于取默认值。

**环境变量（以及作为其 facade 的 flag）：本版本只增加可见性，不改变任何取值。** 2.38 及
之前这些值由各处 `strtol`/`atoi` 宽松解析，写错的值会被悄悄读成别的东西。现在统一经
`src/utils/env_parse.h` 解析：凡是今天能启动的写法，解析结果与此前完全相同；凡是不规范
的写法，每个变量每进程打一行 WARN，指出变量、值和实际采用的值：

- `invalid NAME='值': ...; using N as before ... a later release will not accept this value`
  ——仍按此前的读法采用 N；
- `invalid NAME='值': ...; using the default D`——此前就被忽略而取默认值，现在说明了；
- `NAME='值' clamped to the maximum M`——此前就收敛到上限。

升级后在启动日志里 `grep -E "invalid DFKV_|clamped to the maximum"` 即可找出要清理的项。
**下表这些“带 WARN 兼容接受”的写法将在以后的版本收紧（按字面含义解析或直接拒绝），请
提前清理配置：**

| 写法 | 涉及的变量 | 现在（与此前相同）的结果 | 应改成 |
|---|---|---|---|
| 空串（`NAME=`） | `DFKV_RAM_ACK_HIGH_WATERMARK_PCT`、`DFKV_RAM_RECLAIM_MS` | 读成 `0`：强制 disk-ACK / 关闭 RAM 后台回收 | 要 0 就写 `0`；要默认值就删掉该行 |
| 非数字（`abc`、只有空白） | `DFKV_RAM_ACK_HIGH_WATERMARK_PCT`、`DFKV_RAM_RECLAIM_MS`、`DFKV_RDMA_IDLE_MS`（0 = 关 idle reaper）、`DFKV_SLAB_EVICT_HIGH_PCT`（0 = 关水位驱逐）、`DFKV_SLAB_EVICT_LOW_PCT`、`DFKV_READ_COALESCE_RECUR_MS`、`DFKV_PUT_INFLIGHT_LIMIT` | 读成 `0` | 写数字，或删掉该行 |
| 数字后带别的字符（`30s`、`60%`、`64G`、`1e3`） | 上一行各项，以及 `DFKV_RAM_TIER_BYTES`、`DFKV_RAM_TIER_EXTENT_BYTES`、`DFKV_RAM_TIER_SHARDS`、`DFKV_RAM_FLUSH_THREADS`、`DFKV_SLAB_COLD_STEAL_WINDOW`、`DFKV_RDMA_DEPTH`、`DFKV_SERVER_URING_DEPTH`、`DFKV_READ_COALESCE_TIMEOUT_MS`、`DFKV_RDMA_MAX_PAYLOAD_BYTES`、`DFKV_MDS_MAX_CONNS`、`DFKV_MDS_IO_TIMEOUT_S`、`DFKV_MDS_PROBE_CACHE_MS`、`DFKV_MDS_UNWRITTEN_HOLD_MS`、`DFKV_MDS_ETCD_PROBE_MS` | 只取前面的数字（`64G` = 64 字节） | 写完整的十进制数 |
| 负数（`-1`） | `DFKV_MDS_PROBE_CACHE_MS`、`DFKV_MDS_UNWRITTEN_HOLD_MS`、`DFKV_RDMA_RECV_CHUNK_IDLE_MS`、`DFKV_RDMA_MAX_PAYLOAD_BYTES` | 回绕成极大值后收敛到上限 | 写想要的值 |
| 负数（`-1`） | `DFKV_PUT_INFLIGHT_LIMIT`、`DFKV_RAM_RECLAIM_MS`、`DFKV_RAM_TIER_EXTENT_BYTES`、`DFKV_SLAB_COLD_STEAL_WINDOW` | 回绕成极大值（等于不限 / 几乎不回收 / 单 extent） | 写想要的值 |
| 负数 | `DFKV_RDMA_IDLE_MS` | 等于 `0`（关 idle reaper） | 写 `0` |
| 前导 `+` | 除“首字符必须是数字”的几项外的全部数值项 | 忽略符号 | 去掉 `+` |
| `off` / `false` / `no`（任意大小写） | `DFKV_READ_COALESCE`、`DFKV_RDMA_NUMA`、`DFKV_SERVER_URING`、`DFKV_SLAB_URING_WRITE` | **读成开**：这四项只把 `0` 当关 | 要关写 `0` |
| `00`、`0abc` 等以 `0` 开头的值 | `DFKV_READ_COALESCE`、`DFKV_SLAB_URING_WRITE` | 读成关（看首字符） | 写 `0` |
| `00`、`0abc` | `DFKV_RDMA_NUMA`、`DFKV_SERVER_URING` | 读成开（只认恰好 `0`） | 写 `0` 或 `1` |
| 其他任意词（`enable`、`2`） | 上述四项，以及 `DFKV_RAM_TIER_LOCK`、`DFKV_RAM_TIER_HUGEPAGES`、`DFKV_MDS_ACCEPT_LEGACY` | 读成开 | 写 `1` |
| `ON`、`True`、`2` 等 | `DFKV_RAM_TIER` / `--ram-tier` | **RAM tier 不开**：只认小写 `1/on/true/yes` | 写 `on` |

其余写错的值行为同样不变，只是多了 WARN：负数在用 `strtoull` 读的各项上照旧回绕
（`DFKV_RAM_TIER_BYTES=-1` 仍是 2^64−1 字节的 arena，分配不出来，启动照旧失败；回绕后
落在合法范围内的负数照旧取那个数），在用 `strtol` 读的各项上照旧取默认；
`DFKV_RDMA_RECV_SEGMENT_SIZE` / `DFKV_RDMA_RECV_CHUNK_BYTES` 不是纯数字时照旧取默认值，
WARN 里给出实际采用的字节数；`DFKV_RAM_RECLAIM_MS` 只保留低 32 位（`-1` = 4294967295，
`4294967296` = 0 即关闭回收），照旧，现在会报；超出范围的值照旧取默认或收敛到上限；存储几何与语义类（`DFKV_STORE_ENGINE`、
`DFKV_SLAB_WRITE`、`DFKV_SLAB_PAYLOAD_CRC`、`DFKV_SLAB_GRANULARITY`、`DFKV_SLAB_TABLE_SYNC_MS`、
`DFKV_SLAB_RECLAIM_MS`、`DFKV_SLAB_BALANCE`、`DFKV_SLAB_WARM_RESTART`、`DFKV_DISK_HASH_WEIGHT`、tenant quota、
`DFKV_RAM_TIER_NUMA`）和 `DFKV_MDS_REGISTRATION_TIMEOUT_MS` 写错时照旧报错退出。

**直接解析的命令行参数：此前被悄悄截断或回绕的值现在报错退出（exit 2）。** `--port`、
`--rdma-port`、`--metrics-port`、`dfkv_mds --listen` 必须在 0..65535（此前 `--port 70000`
实际监听 4464；可选监听项显式写 `-1` 仍表示不开）；`--cap`、`--max-msg` 不接受负数（此前
`--cap -1` 回绕成 2^64−1）；放不进 int 的整数不再回绕；`dfkvctl --stat-port` 必须是
1..65535（此前非数字被读成“用成员自己的端口”）。`--mds` 缺 `--id`/`--advertise` 在打开任何
盘之前就报错（exit 1，此前在盘和 RAM tier 都起来之后）。未给 `--dir`/`--cap` 仍用默认值
启动，但会打 WARN。仓库自带的 Dockerfile、`deploy/` 脚本和示例 unit 里的参数值都不受影响。

#### a. 核心必填/常用（`dfkv_server --help` 全集）

| flag/env 孪生 | 默认 | 说明 |
|---|---|---|
| `--dir <paths>` | —（必填）逗号分隔多盘 | 每节点 NVMe 数据目录，逗号加多个自动 Ketama 分盘；`--cap` 是**总容量**，按盘均分 |
| `--cap <bytes>` | —（必填） | 节点总容量（LRU 超限触发自裁） |
| `--port <p>` | 0(临时） | TCP bootstrap+数据端口；0=随机 |
| `--rdma-port <p>` | —（RDMA build 必填） | RDMA bootstrap 端口（含 RDMA 时开启） |
| `--rdma-dev` | first ACTIVE 本地 | server RDMA 白名单（逗号）；留空保持 ACTIVE-only 自动发现。显式列表按首次出现去重，present DOWN 也须初始化并监控；missing/open/query/任一 anchor 或 MR 失败则拒绝启动 |
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
| `DFKV_SLAB_PAYLOAD_CRC` | `1` | slab 值校验：PUT 把值的 CRC32C 记进 slot record，整值读取（GET、RDMA 读）时校验，不符则删除该 key、按未命中回答并计入 `dfkv_slab_payload_crc_failures_total`。只读一段的请求不校验。校验发生在读取时：落盘前被 kill、warm restart 后保留下来的 record，在第一次整值读取之前 EXISTS 仍报存在。代价（单写者、1 MiB 值、NVMe 单元级实测）：PUT 中位延迟多约 60–70 µs（约 +17%，单写者吞吐约 −12%），GET 多约 15–40 µs；多个写者时瓶颈在盘，影响更小。设 `0`：PUT 不再记录校验值（record 与旧版本写的相同），任何读取都不校验，包括开着时写下的 record；关闭期间写下的 record 重新打开后也一直不校验，直到被重写。只接受 `0`/`1`，其他值拒绝启动 |
| `--ram-tier` / `DFKV_RAM_TIER` | `off` | RAM 热层（写穿+RDMA zero-copy GET） |
| `--ram-tier-bytes` / `DFKV_RAM_TIER_BYTES` | `16 GiB` | arena 预算（启动时预触页并锁在内存里） |
| `--ram-tier-numa` / `DFKV_RAM_TIER_NUMA` | `interleave` | NUMA 策略：`interleave`/`off`（不接 node id） |
| `--ram-tier-shards` / `DFKV_RAM_TIER_SHARDS` | `8`, 硬上限 64 | arena 锁分片数（大 arena ≥100 GiB 建议 16） |
| `--slab-granularity` / `DFKV_SLAB_GRANULARITY` | `1 MiB` | slab slot 量子；现有目录 geometry 不符启动拒绝；小对象负载见 §3a-1 |
| `--put-inflight-limit` / `DFKV_PUT_INFLIGHT_LIMIT` | `0`=关 | 前台 PUT 接纳上限（RAM 与直接盘写）；超限在接纳前返回 kCacheFull，后台 flush 不受限 |
| `--tcp-max-conns` / `DFKV_TCP_MAX_CONNS` | `512`, 硬上限 4096 | cache TCP handler 上限；超限 accept 恒拒 |
| `--tcp-io-timeout-s` / `DFKV_TCP_IO_TIMEOUT_S` | `60`, 硬上限 3600 | per-syscall RCVTIMEO（秒） |
| `DFKV_RDMA_MAX_PENDING_HANDSHAKES` | `4096`，`0`=不限 | RDMA bootstrap 监听上“已 accept、bootstrap 帧尚未收齐”的连接数上限；超限的新连接不建线程直接关闭并计入 `dfkv_rdma_rejected_connections_total`。只计还没发完帧的连接，已建立的 RDMA 连接数不受它限制。客户端 connect 之后要先建好自己的 QP 才发帧，所以名额从 accept 一直占到客户端 QP 创建完成；大批客户端同时重连（例如 server 重启后）时，在途数约等于 accept 速率 × 客户端建 QP 的耗时。被拒的客户端表现为 bootstrap 失败并重试。此上限用来挡只连接不发帧的对端；`dfkv_rdma_rejected_connections_total` 在集中重连期间增长说明上限偏小，应调大。bootstrap 帧须在 accept 后 10 s 内收齐（absolute deadline，不可配） |
| `--rdma-depth` / `DFKV_RDMA_DEPTH` | `4` | server ceiling；client scalar=1，batch按实际window选择depth class后再协商取min |
| `--rdma-numa` / `DFKV_RDMA_NUMA` | `0` | NUMA-aware rail choice（off/1） |
| `--rdma-idle-ms` / `DFKV_RDMA_IDLE_MS` | — | idle connection reaper tick |
| `--rdma-op-timeout-ms` / `DFKV_RDMA_OP_TIMEOUT_MS` | `5000` | client：单个操作的整体期限，从操作开始算：建连（准入、budget 等待、能力探测、拨号与 bootstrap 读）、GET 的请求、READ 和释放、以及重试都只用剩下的时间。批量操作（`DFKV_RDMA_BATCH_OP_TIMEOUT_MS`）每次尝试的建连受它约束，各窗口仍各有自己的期限。在被期限截短的时限上超时的建连不记对端不可达（被拒绝的仍然记），被期限截短而超时的 bootstrap 读和 GET 释放不冷却 rail（server 的 receive pool 紧张时 bootstrap 回复可能等 5 s），所以按默认值（操作期限 5 s 小于 `DFKV_RDMA_IO_MS` 的 10 s）bootstrap 上的沉默从不冷却 rail：操作照样失败，由节点级健康（peer cooldown）计入，计入 `dfkv_rdma_client_op_deadline_expired_total`；拒绝确认拨号也不超出期限 |
| `--server-uring` / `DFKV_SERVER_URING` | `0` | io_uring async-GET（编须 `-DDFKV_WITH_URING`） |
| `--server-uring-depth` / `DFKV_SERVER_URING_DEPTH` | — | uring SQ 深度 |
| `--ram-flush-threads` / `DFKV_RAM_FLUSH_THREADS` | 默认 4×盘数， 钳至 shard 数上限 16；至少 shard 数，多盘时至少 2×shard 数 | RAM tier flush 线程 |
| `--slab-table-sync-ms` / `DFKV_SLAB_TABLE_SYNC_MS` | `boot` 下 `100 ms`，`auto` 下 `5 s`，`off` 下 `0`（关） | 运行期 `slots.tbl` 周期同步节奏；`boot` 的恢复靠它，`auto` 的 5 s 用来及时发现表的回写错误 |
| `DFKV_SLAB_WARM_RESTART` | `off`（`auto` / `boot`） | 非干净退出后保留还是清零 slab 缓存 |
| `--slab-reclaim-ms` / `DFKV_SLAB_RECLAIM_MS` | `50 ms`, `0`=关 | slab 后台回收 tick |
| `--ram-reclaim-ms` / `DFKV_RAM_RECLAIM_MS` | `10 ms`, `0`=关 | RAM tier 后台回收 tick |
| `--log` / `DFKV_LOG` | `INFO` | INFO/DEBUG/WARN/ERROR，大小写均可；不认识的词按 INFO 并在 stderr 提示 |
| `--max-msg`（或 `DFKV_RDMA_MAX_PAYLOAD_BYTES`） | `32 MiB` | 单笔 payload 硬上限（RDMA 与 TCP 数据路径同受此限） |
| `--version, -V` / `--help` | — | 打印版本/帮助并退出 |

#### b. RDMA 传输面（server/client 读取方见各项说明）

| env | 默认 | 说明 |
|---|---|---|
| `DFKV_RDMA_RECV_SEGMENT_SIZE` | `128 GiB` | server receive-pool hard budget（含单 chunk staging reserve）；不再启动期全量分配 |
| `DFKV_RDMA_RECV_CHUNK_BYTES` | `256 MiB` | server 启动与增量提交粒度；有足够 QP 余量时首 chunk 专供操作级暂存 |
| `DFKV_RDMA_RECV_CHUNK_IDLE_MS` | `60000` | 空闲非初始chunk返还延迟；`0`关闭缩容 |
| `DFKV_RDMA_RECV_HUGEPAGES` | `on` | receive pool 的 chunk（≥2 MiB 时）申请透明大页，和 RAM 层 arena 一样（每个 chunk 一段单独的 `mmap` 加 `madvise(MADV_HUGEPAGE)`）：`/sys/kernel/mm/transparent_hugepage/enabled` 为 `madvise` 或 `always` 时生效，为 `never` 时不起作用、chunk 仍是 4K 页。从盘读的 GET 和 leased PUT 每次都要为本次操作注册一段 chunk 内存，大页上注册和注销快得多：实测 4 MiB 约 0.05 ms 对 0.3 ms，64 MiB 约 0.4 ms 对 2.4 ms；64 KiB 以下差别不大（每次约 20–40 µs 的系统调用本身） |
| `DFKV_RDMA_CONNECTION_MIN_BLOCK_BYTES` | `256 KiB` | client adaptive data-QP 最小 class；实际对象向上取 power-of-two |
| `DFKV_RDMA_INLINE_PUT_MAX_BYTES` | `4 MiB` | client 超阈 PUT 使用操作级 lease；`0`/空值禁用，不改变业务对象上限 |
| `DFKV_RDMA_SPIN_US` | `0`（关），上限 10000 | client：等 completion 时先轮询 CQ 这么久再挂起等事件（事件模式），轮询时间算在本次等待的期限内。对小对象、看重延迟的部署建议设 `50`：单个 GET 是三次往返，每次挂起再唤醒约 30–50 µs，实测 64 KiB GET 中位延迟约 140 µs → 约 50 µs，与 `DFKV_RDMA_BUSY_POLL` 相当。代价是 CPU：server 比轮询时间慢时，每个 GET 最多空转约 150 µs（三次各 50 µs），每秒 1 万次 GET 约占 1.5 个核；在有 CPU 配额（cgroup quota）的容器里会更早被限流 |
| `DFKV_RDMA_CONNECT_MS` | `3000` | client TCP bootstrap **建连**超时（毫秒），不覆盖整个 QP 初始化或后续读写；同时不超过操作剩下的期限 |
| `DFKV_RDMA_IO_MS` | `10000` | client TCP bootstrap 帧读写超时（毫秒），同时不超过操作剩下的期限（`DFKV_RDMA_OP_TIMEOUT_MS`）；RDMA completion 另由后者限定 |
| `DFKV_RDMA_BATCH_OP_TIMEOUT_MS` | 0=跟随 RDMA_OP | client：multi-item Cache/Range/Exist、SG 窗口总期限 |
| `DFKV_RDMA_POOL_MAX` | 未设：常驻基线 `8`、data 突发上限 `64` | client：scalar/SG 共用 data pool；超过基线的空闲连接在最后一次应用使用 60 秒后退休，keepalive 不续期。显式正整数同时固定基线与上限；control 仍取基线，全部受进程资源预算约束 |
| `DFKV_RDMA_RAIL_CREDITS` | 默认 `64`, 硬上限 4096 | client：每 rail outstanding request credits |
| `DFKV_RDMA_RAIL_TIERS` | unset = homogeneous | client：异构带宽 tier `a|b;c`，只在最高健康 tier 内选轨，不因 credit/quarantine 溢出到低 tier |
| `DFKV_RDMA_PRIMARY_DEV` | unset | client：必须属于 `DFKV_RDMA_DEV`；所有 operation 优先该 rail，仅在不可准入/失败时使用其余 configured rail |
| `DFKV_RDMA_RAIL_BACKPRESSURE_MS` | `200`（不超过 `DFKV_RDMA_OP_TIMEOUT_MS`） | client：一个逻辑操作等待 rail credit 的总预算，按操作计而不是按尝试计：其所有尝试（重试、换 rail、逐 rail 选择）共用这 200 ms，从该操作第一次准入算起；用完返回 `kResourceExhausted`。等待期间 NUMA 本地与非本地 rail 谁先空出就用谁，本地仍优先（此前默认每次准入各等 10 ms，只等本地 rail） |
| `DFKV_RDMA_RAIL_COOLDOWN_MS` | `5000` | client-local rail quarantine 冷却期；冷却期间不准入。期满后由后台线程（keepalive/端口检查线程，默认开启）在该 rail 上对已连接过的节点（每次最多试 4 个）发一次 keepalive 作为 recovery probe（建连最长 `DFKV_RDMA_CONNECT_MS`，期间该线程的端口检查和 keepalive 顺延），应用操作只在健康 rail 都没有 credit 时才落到它上面；后台线程没有运行时（keepalive、端口检查都关且无 burst 池）仍由下一个真实 operation 作 probe |
| `DFKV_RDMA_RAIL_ERROR_THRESHOLD` | `3` | client：连续 `kRailFailure` 达到该值才 quarantine；peer/endpoint、admission、timeout（无 local-error WC）、cancel 和 invalid input 不计入 |
| client-local rail attempts | 固定 `2`（不可配置） | GET/batch/SG 第二次只重试未完成 item/window；PUT 仅在确认 request 未 post、或所用 pooled QP 自出池后无任何 completion 成功（stale）时重发，其余 post 后 ambiguous failure 不 replay |
| `DFKV_RDMA_RAIL_LATENCY_WEIGHT` | — | client：rail selection 的延迟 EWMA 权重。样本从连接就绪算起（不含建连），超过 64 KiB 的操作按每 64 KiB 折算（固定开销也被折算，承载大对象的 rail 会显得略快）；一条 rail 每 1 s 没有新样本 EWMA 减半，避免一次慢样本让它长期不被选中 |
| `DFKV_RDMA_RAIL_ERROR_PENALTY_US` | — | client：rail selection 的错误分数惩罚微秒 |
| `DFKV_RDMA_IDLE_MS` | `600000` | server 常规空闲 QP 超时（毫秒；`0` 禁用）；接收池压力回收另须保护未完成的 reply/READ。**至少设为 client `DFKV_RDMA_KEEPALIVE_MS`（默认 15000）的 3 倍**：client 只靠 keepalive 让池里的 QP 在 server 侧保持活跃，留一拍余量时 client 被限流或暂停一下 server 就会回收，client 下一次用到这条 QP 要等 RC 重试耗尽（实测约 3.7 s）才重连。client 关掉 keepalive（`0`）时，server 每次空闲回收、每次重启之后，client 第一次用到池里对应的 QP 都要付这 3.7 s |
| `DFKV_RDMA_PIPELINED_READS` | `0` | client 开关：同一 QP 上最多 16 个 RDMA READ 同时在途（默认每次只有一个在途，GET 窗口里的 READ 被网卡串行化）。只对在探测回答里声明了 READ 深度的 server 生效（本版本起的 server 在每条 rail 的网卡都支持 16 个时才声明；server 无条件接受，对不开的 client 没有影响），而且要 server 在握手应答里对这条连接确认后才提高深度，否则仍是一次一个（超过对端 QP 接受的数目会让 QP 因协议错误被关闭）。client 缓存了探测结果、而对端在同一身份下换成更早的版本时，会被对方关掉一次握手（当作对端忙，暂停约 20 ms 后重新探测），不会出现失败的 READ。实测单连接 4 × 1 MiB 的 READ 窗口 179 → 127 µs（23 → 33 GB/s），16 × 64 KiB 101 → 38 µs。计入 `dfkv_rdma_client_pipelined_read_conns_total`。server 升级后再开。path MTU 取本端端口的 active MTU，不与对端协商（`DFKV_LOG=debug` 时每条连接打印一次） |
| `DFKV_RDMA_SESSION_FD` | `0` | 会话通道（server 和 client 各自的开关，两边都开才生效）：建连用的 TCP socket 不再在握手后关闭，作为这条 RDMA 连接的会话通道，任一侧结束连接都会关掉它，另一侧立刻看到 EOF。server 因空闲回收、接收池压力驱逐、协议错误、授权超时或停止而结束连接时，client 复用池里这条连接之前先非阻塞探测（`MSG_PEEK`），发现已关就直接丢弃重连（实测复用第一次操作从约 3.7 s 降到约 10 ms，计入 `dfkv_rdma_client_session_closed_reaps_total`）；client 丢弃连接或进程退出时 server 立刻结束对应连接，不再拖到 `DFKV_RDMA_IDLE_MS`（`dfkv_rdma_session_peer_closed_total`）。只在 server 的探测回答里声明了这项能力时 client 才请求，所以新 client 对旧 server、旧 client 对新 server 都照旧工作；server 和 client 都升级后再开。client 的 keepalive 也先探测，跳过并丢弃已关的连接。探测之后、操作之前恰好被 server 关掉的连接，仍要等旧的 RC 重试超时（窗口很窄）。代价：client 每条池化连接多一个文件描述符（已有一个 completion channel），开启时 client 把 `RLIMIT_NOFILE` 软限制提到硬限制；连接很多时确认硬限制够用（按节点数 ×（8 + 最多 64 个突发连接）× 2 估算）。server 每条开了会话的连接也多占一个文件描述符，`LimitNOFILE` 要覆盖它的连接数，否则 `accept()` 会更早遇到 EMFILE。该 socket 创建时即 close-on-exec，但普通 `fork()` 出来的子进程（Python multiprocessing 的 "fork" 方式）在退出前一直持有副本，这些连接 server 看不到关闭，仍靠空闲超时。客户端主机整机宕机不会发 FIN，这种情况也靠空闲超时。client 和 server 之间的建连路径上不能有在握手后就关掉 TCP 连接的代理：server 会把它当成 client 已断开，立刻结束每条连接 |
| `DFKV_RDMA_STOP_DRAIN_MS` | `2000` | 停止时让还持有授权（GET 已发 READY、PUT 已发租约）的连接先做完的上限（毫秒；`0` 不等）。期间新的 GET 答 miss、新的 leased PUT 答 `kCacheFull`（计入 `dfkv_rdma_stop_grants_refused_total`），只等已发出的授权；没有授权的连接立即关闭；握手中的连接立即断开，不再等 bootstrap 的 10 s 截止。到期仍未做完的计入 `dfkv_rdma_stop_drains_cut_total`。它算在停止预算里：和 `DFKV_RAM_SHUTDOWN_DRAIN_TIMEOUT_S`、停请求、各盘同步一起要小于 `TimeoutStopSec` |
| `DFKV_RDMA_GRANT_TIMEOUT_MS` | `60000` | 持有授权（GET 已发 READY 未 RELEASE、PUT 已发租约未写入、多窗口 PUT 未到下一窗）而 server 自己没有活（没有在途的读盘或回复）的连接最多静默多久（毫秒；`0` 不设上限），从它最后一次收到 completion 和最后一次发出授权两者中较晚的那个算起：一批 GET 的时钟从它的最后一个 READY 开始，前面的 READY 等盘多久都不算。超过就关掉这条连接（先销毁 QP 和 MR，再放 pin 和暂存区），计入 `dfkv_rdma_grant_expiries_total`。client 崩溃或放弃后，这些 pin 不再拖到 `DFKV_RDMA_IDLE_MS`。client 活着、连接一直有流量但漏发某个 RELEASE 的情况不在此列（仍要等连接空闲）。client 的 `DFKV_RDMA_OP_TIMEOUT_MS`（或设了的 `DFKV_RDMA_BATCH_OP_TIMEOUT_MS`）超过它时，一次 READ 或写入本身超过它的连接会被切断；`DFKV_RDMA_OP_TIMEOUT_MS=0` 时把它也设成 `0` |
| `DFKV_RDMA_SERVER_MW` | `0` | server 开关：GET 授权用内存窗口（memory window，type 2B），不再每次注册、注销一个 MR。只用于 RAM 层（arena）里的值（没开 RAM 层时开关不起作用，连接不分配窗口，启动日志会说明）：arena 的每个 extent 在每条 rail 上只注册一次（只允许本地写和绑定窗口，没有远端权限），每个 GET 在本连接的 QP 上绑定一个窗口、RELEASE 时作废它（各约 3.5 µs；一对 MR 注册注销约 55 µs，4 MiB 约 220 µs）。实测单个 GET：server 为每个授权花的时间 15–30 → 1–1.5 µs；RAM 层 64 KiB 的中位延迟 221 → 146 µs、1 MiB 220 → 198 µs，4 MiB 反而 424 → 508 µs（RELEASE 的回复要排在作废之后）。要从盘暂存的 GET 不用窗口，照旧逐次 MR（盘刚 DMA 写进的内存，经 DMA 之前就有的注册去 READ 会慢很多，经 DMA 之后注册的 MR 不慢；见设计文档第 13 节，RDS-29）。线上协议不变，新旧 client 都适用，与升级顺序无关；只 server 设置。只在网卡支持 type-2B 窗口的 rail 上生效（启动日志每条 rail 一行，`dfkv_rdma_server_mw_mode{dev}`；NVIDIA 只在 mlx5 物理功能上提供，VF 照旧用逐次 MR）。窗口作废之后，范围要再隔离一段时间才复用（下一行）。**剩余风险**：只涉及机密性，而且只针对违反协议的 client（在自己的 RELEASE 之后还读）；不会造成数据损坏。在 mlx5 上，作废之前已被网卡接收的 READ、或这个 READ 的重传，在该 client 的传输重试期限内（默认约 3.7 s，`DFKV_RDMA_QP_TIMEOUT` × `DFKV_RDMA_QP_RETRY_CNT`）可能读到已被复用的范围，隔离期覆盖不了这一点（Soft-RoCE 会对每个响应包重新检查 rkey，没有这个问题；mlx5 是否重新检查尚未确认）。接受不了的部署不要开。**默认值**：在真实硬件上确认 mlx5 对重传的 READ 是否重新检查 R_Key（厂商说明或可靠的测试）之前，保持默认关。**并发下的吞吐**（2.35.7 的 `dfkv_bench`，4 个客户端线程，开关交替跑多轮取中位数）：数据在 RAM 层时窗口持平或更快（64 KiB、每批 8 个：GET 26600 → 42700 ops/s；256 KiB 单键持平或略好）；GET 要从盘暂存时与关闭持平（256 KiB 单键：5040 → 5390 ops/s，15 轮）。早先窗口也覆盖接收 chunk 时，从盘暂存的 GET 开启后变慢（同一测法 4960 → 3370 ops/s），原因（RDS-29，裸 verbs 实测）：网卡读一段刚被 NVMe DMA 写入的内存，经写入之前就有的注册（静态 MR，或在更早注册的区域上绑定的窗口）去读明显慢，经写入之后才注册的 MR 去读不慢（256 KiB READ 中位数约 115–225 µs 对 25–30 µs）；CPU 写入也有这个现象，GPU 拷贝写入没有；刷 CPU cache、等 10 ms 都不能恢复；设备层面的原因不明，要问网卡厂商（数据见设计文档第 13 节）。所以现在只对 RAM 层用窗口。早先记录的"成批 GET 吞吐减半"是 RAM 层装不下测试数据、GET 走了盘暂存所致。arena 按 extent 注册：`LimitMEMLOCK` 要能覆盖整个 arena 一次（随包的 unit 是 `infinity`）。退回逐次 MR 的情况都有计数：`dfkv_rdma_server_mw_fallback_grants_total`（arena extent 还没注册好，注册在后台、第一次用到时开始；从盘暂存的 GET 不算）、`_quarantine_fallbacks_total`、`_fallback_connections_total`（窗口数超过网卡的 `max_mw`，启动时若可能不够会打 WARN） |
| `DFKV_RDMA_SERVER_MW_QUARANTINE_MS` | `5` | 开了 `DFKV_RDMA_SERVER_MW` 时，作废后的范围最少隔离多久（毫秒；只能调高，低于 5 按 5）。**实际隔离期统一为 `DFKV_RDMA_SERVER_MW_QUARANTINE_MAX_MS`（不小于这个下限）**，不使用可能会造成不足保护的授权时 rail 快照：本 grant 授权后，其他连接还可能授予更多读取，冻结当时数字并不能提供未来时间上界。rail 估算（2 × 当前在途 GET 字节 ÷ 最慢 rail 速率 + 1 ms）仅在授权时选择窗口仍是逐次 MR，不参与决定隔离时长。实测 400G 端口、16 个进程各读 4 MiB 时 READ 的 p99.99 为 1.4 ms；上限为保守时间预算，不是声称任何设备一定按要求停止重传（mlx5 的限制见上一行）。隔离期内范围（staging 租约或 RAM arena 的 pin）不复用；指标 `dfkv_rdma_server_mw_quarantine_bytes` 只统计实际隔离中的范围。授权时独占预留计入容量上限，随后转正式隔离，避免并发超额度。 |
| `DFKV_RDMA_SERVER_MW_QUARANTINE_MAX_MS` | `50` | 一个 GET 所需隔离期超过它时，这个 GET 改用逐次 MR（计入 `dfkv_rdma_server_mw_quarantine_fallbacks_total`）。不低于上一行；对已准入的 grant 不是 close 时截短隔离期的工具 |
| `DFKV_RDMA_SERVER_MW_QUARANTINE_BYTES` | 接收池上限的 1/8 | 隔离容量上限（arena pin 和暂存 lease 实际占用的字节，含 slab 槽 padding，不按请求的 payload 长度假算）；已绑定、尚未正式隔离的 grant 预留额与实际隔离共同占用，不会因并发检查竞态瞬时超限。改为逐 grant RAM 直接读取时会提前归还暂存 lease，预留仍保留同一数量，防止授权预留在撤销时 mismatch。到了上限或所需隔离期过长时，新的 GET 改用逐次 MR，隔离区不会挡住授权，也不会把接收池占满 |
| `DFKV_RDMA_SERVER_MW_ARENA_BYTES` | `0`（整个 arena） | 每条 rail 最多为窗口注册多少 RAM arena 字节；超出部分的 extent 不注册，其中的 GET 用逐次 MR |
| `DFKV_RDMA_SERVER_MW_LINK_GBPS` | `0`（按端口） | 计算隔离期用的链路速率下限（Gbit/s）；比各 rail 端口的实际速率低时用它。端口速率查不到时按 10 Gbit/s |

> **RDMA 设备重置与 client 的设备上下文**：client transport 从每条 rail 第一次建连起一直持有该网卡的设备上下文（context 和 PD），直到 transport 销毁，连接池清空也不关闭（省去每次重开约 11 ms）。网卡被拔出或复位后，该 rail 上新建 endpoint 失败（`ENODEV`/`EIO`）时 transport 放下这个持有，剩下的连接都结束后上下文关闭，之后的建连重新打开设备。调用过 `RegisterMemory` 的 transport（引擎接入的常见情况）另有每条 rail 一个注册锚点，同样一直持有设备上下文，这一版之前就是如此：设备复位后它仍握着失效的上下文，这条 rail 要重启 client 进程才能恢复。打开 `DFKV_RDMA_DEVICE_REBUILD=1`（client 侧开关，默认关）后不必重启：收到网卡的 `IBV_EVENT_DEVICE_FATAL`，或这条 rail 上新建 endpoint 失败（`ENODEV`/`EIO`）时，这个设备名的绑定作废（下次建连打开新的上下文），这条 rail 退出该 transport 的选择，后台线程在新上下文上重建注册锚点、把声明过的内存逐块重新注册，成功后 rail 回到选择（设备还没回来时按 1 s 起、最长 30 s 的间隔重试）。重建期间这条 rail 上进行中的操作照旧按 rail 故障处理（可重放的换 rail 重试，已发出的 PUT 不重放）；只有这一条 rail 的 client 在重建完成前操作失败。开关打开时，若释放失败的上下文已被证明失效（收到过 DEVICE_FATAL，或确认查询返回 `ENODEV`，或返回 `EIO` 且设备已不在设备列表/sysfs 节点已换），释放失败只计数（`dfkv_rdma_client_dead_release_failures_total`）不再中止进程；其他情况照旧中止。开关关闭时只计数这些信号（`dfkv_rdma_client_async_events_total`、`dfkv_rdma_client_device_suspect_total`），其他行为不变。server 侧不重建：设备复位后要重启 server（热重启保留磁盘数据）。重建路径目前只通过测试桩验证过（模拟 DEVICE_FATAL 和 `ENODEV`）：Soft-RoCE 不能让上下文失效（`rdma link delete` 会一直等到所有上下文关闭），真实网卡又不能拿来复位，这也是开关默认关闭的原因之一。

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
| `DFKV_SLAB_EVICT_HIGH_PCT` / `DFKV_SLAB_EVICT_LOW_PCT` | `92` / `88`, high=0 关 | high crossing latch，后台持续 drain 到 low |
| `DFKV_SLAB_EVICT_HEADROOM_BYTES` / `DFKV_SLAB_EVICT_BAND_BYTES` | 不设（用百分比） / 余量的 1/4 | 绝对余量：high = 容量 − 余量，low = high − band；优先于百分比；band ≥ 1 个 extent，余量 ≥ 2 band（§3） |
| `DFKV_SLAB_EVICT_MAX_VICTIMS_PER_TICK` | `256` | 每盘每 tick 最多移除的 slot 数；最多扫描该值的 4 倍，不能据此保证尾延迟不变 |
| `DFKV_SLAB_COLD_STEAL_WINDOW` | — | 小受害者选择窗口（跨 size class 强占） |
| `DFKV_SLAB_URING_WRITE` | — | slab io_uring 异步 PUT（需要 `-DDFKV_WITH_URING`） |
| `DFKV_SLAB_RECLAIM_MS` | `50` | 背景预回收 tick（见 §3） |
| `DFKV_SLAB_BALANCE` | `0` | `1`：每个回收 tick 在各尺寸类之间按“下一个要淘汰的值有多久没被用过”（tail age）挪 extent——在都在淘汰的类里，tail 最老的类把一个最久未用的 extent 让给 tail 最年轻的类，前者超过后者 1.1 倍才挪，每 tick 最多一个。解决的是负载里各尺寸占比变了（例如大对象占比从三分之一升到八成）而各类都仍在写时，空间停在旧比例、大类只留得住少量新写入的问题（单元测量：大类近一个容量的写入只留 72–77%，小类 100%；开启后三类相同）。不挪含正在读写的槽或 RETAIN 保护值的 extent；挪动走现有的 extent 重绑定：先写清表再丢值（STO-11：由页缓存保住，供 warm restart 使用），warm restart 安全。每次挪动在 store 锁内淘汰整个 extent 的值：1 GiB extent、128 KiB 槽时约 0.9 ms，`dfkv_slab_balance_max_hold_us` 记录实际值。只用于磁盘 slab，RAM 层不受影响。只接受 `0`/`1`，其他值拒绝启动。设计与测量见 docs/design/slab-cross-class-eviction.md |
| `DFKV_SLAB_GRANULARITY` | `1 MiB` | slot 量子 |

#### e. RAM 热层细节调优

| env | 默认 | 说明 |
|---|---|---|
| `DFKV_RAM_TIER_EXTENT_BYTES` | — | arena extent 大小（通常保留默认；与 `DFKV_RAM_TIER_BYTES` 比例决定并发度） |
| `DFKV_RAM_TIER_LARGE_RESERVE_BYTES` | `0`（不保留） | 超大对象（超出 arena extent）的 dedicated allocation 预算。默认不保留：超出 extent 的值不进 RAM，直接写盘、从盘读（早于此的版本默认给它们留两个 extent）；设一个预算即恢复为进 RAM。默认 `--max-msg` 等于 extent，到不了这条路径；只有调大 `--max-msg` 时才有意义。设了之后预算满时按进入顺序淘汰最老的（只看这类对象，不扫整个索引） |
| `DFKV_RAM_TIER_SHARDS` | `8`, 硬上限 64 | 锁分片 |
| `DFKV_RAM_TIER_NUMA` | `interleave` | interleave / off |
| `DFKV_RAM_TIER_LOCK` | `on` | arena 是否锁在内存里（`mlock2`）；锁不上只告警，`dfkv_ram_arena_locked` 反映实际结果 |
| `DFKV_RAM_TIER_HUGEPAGES` | `on` | arena 是否申请透明大页（`madvise(MADV_HUGEPAGE)`）。系统的 `transparent_hugepage/enabled` 是 `madvise` 或 `always` 时生效，是 `never` 时等于没开；内核给不出大页的部分自动退回 4K 页。设成 `off` 只是不申请，页大小由系统设置决定：`madvise` 下是 4K 页，`always` 下仍然是大页。实际大页承载量看启动日志和 `dfkv_ram_arena_hugepage_bytes` |
| `DFKV_RAM_FLUSH_THREADS` | `4×盘数`，钳至 shard 上限 16；至少 shard 数，多盘时至少 2×shard 数 | flush 线程 |
| `DFKV_RAM_PUT_WAIT_MS` | `3000`，上限 3600000，`0` 表示不限 | 一个 PUT 等自己这个 key 的属主值的最长时间：disk-ACK 下每个被接纳的 PUT（首写者和重复 PUT）都等这个值刷盘；RAM-ACK 下重复 PUT 只等到首写者的值在 RAM 里可见。时间从这个 PUT 所在的请求批次到达时算起：一条 RDMA 连接按顺序处理客户端一个窗口里的请求，客户端给整个窗口一个截止时间，所以同一窗口里要等待的 PUT 共用这段时间，而不是每个重新计时。等待期间不占用 PUT 的 key 分片锁。到时回 `kCacheFull`（这次请求没看到落盘，可重试；已在 RAM 里的值照常继续刷盘，客户端不会因此把节点判坏）。要小于客户端的操作超时（`DFKV_RDMA_OP_TIMEOUT_MS`，默认 5000；批量窗口看 `DFKV_RDMA_BATCH_OP_TIMEOUT_MS`）：客户端先超时会按 IO 错误冷却这个节点。必须是纯数字，否则按默认值处理并打一条 WARN；超过上限也按默认值 |
| `DFKV_RAM_ACK_BACKPRESSURE_WAIT_MS` | `0`（立即拒绝，与以前相同），上限 60000 | RAM-ACK 下，未落盘数据达到水位（`DFKV_RAM_ACK_HIGH_WATERMARK_PCT`，默认 80%）时，新 key 的 PUT 最多等多久让刷盘把未落盘数据降下来，再决定接纳还是回 `kCacheFull`；也不超过 `DFKV_RAM_PUT_WAIT_MS`（从请求批次到达算起）。等待时不占 key 的锁，每刷完一个值就重试一次接纳；停止时立即结束等待。为 0 时被拒绝的 PUT 由客户端按"没存上"处理，连接器不会重试，之后读取 miss。**建议设 50**：实测 256 MiB 的 RAM 层先写入 250 MiB、再 4 线程 × 250 个 256 KiB 新 key（slab 引擎、direct 写一块 NVMe），不等待时三轮里 1000 个分别有 625、527、586 个被拒；等待上限 50 ms 时 0 个被拒，p99 4.7–6.9 ms、最长 8.8 ms（刷盘几十毫秒就能清空积压）。代价：等待期间这个服务线程不处理同一连接后面的请求。计入 `dfkv_ram_ack_backpressure_waits_total`、`_wait_timeouts_total`、`_wait_microseconds_total` |
| `DFKV_RAM_PROMOTE` | `first` | 盘上整值冷读什么时候装进 RAM：`first` 每次都装（以前的行为）；`second` 只在这个 key 在 `DFKV_RAM_PROMOTE_WINDOW_MS` 内已经被读过一次时才装，第一次读只记下 key（指纹，不存数据），不预留也不碰 RAM。适合一次性扫描和反复读的热数据混在一起的负载：扫描不再把热数据挤出 RAM。代价是每个 key 至少多一次盘读（第二次读才装进去，第三次起才从 RAM 读）。其他值告警后按 `first` 处理 |
| `DFKV_RAM_PROMOTE_WINDOW_MS` | `30000`，范围 1–86400000 | 只在 `second` 下读取：两次读间隔不超过这个值才算"又读了一次"。按负载里同一个 key 两次读之间的预期间隔来定，取比它稍大的值（例如同一前缀约 10 秒复用一次，就用 15000–30000）：比间隔小，热数据永远装不进 RAM；太大，扫描里偶尔被读两次的 key 也会装进去。不跟着保留时长（retention TTL）走：那是另一回事。记录的 key 最多 262144 个（约 36 MB），满了先忘最久没读的。`dfkv_ram_promote_deferred_total` 是因此没装的读，`dfkv_ram_recent_reads` 是当前记着的 key 数；非法值（0、非数字）告警后用默认值 |
| `DFKV_RAM_SHUTDOWN_DRAIN_TIMEOUT_S` | `30`，上限 3600，`0` 表示不等 | 停止时给 RAM 层积压刷盘的时限，从开始停止算起。到时还只在内存里的对象直接丢弃（缓存语义，日志里有数量），然后各盘照常同步并写 clean。要小于 `TimeoutStopSec` 减去停请求和各盘同步的时间；默认值按 systemd 默认的 90 秒停止超时取的，`TimeoutStopSec` 调大了就一起调大。非法值（非数字、带单位）告警后用默认值 |

> **大对象为主时调大 extent。** arena 按 extent（默认 32 MiB）分给各尺寸类，一个
> extent 只放整数个同类 slot，放不下的尾部空着；slot 本身还按尺寸格点向上取整
> （最多约 25%）。对象较大时尾部占比很高：一个 17 MiB 的值占约 19 MiB 的 slot，
> 32 MiB 的 extent 只放得下一个。单元级实测（768 MiB arena、单一值大小，前一个数
> 是不开后台回收，后一个是默认 10 ms 回收，回收会多留几个空 slot）中有效载荷占
> arena 的比例：
>
> | 值大小 | 32 MiB extent | 128 MiB extent |
> |---|---|---|
> | 约 1 MiB | 95.6% / 88.3% | 95.6% / 87.5% |
> | 10 MiB | 62.5% / 58.6% | 85.9% / 72.9% |
> | 17 MiB | 53.1% / 48.7% | 79.7% / 70.8% |
>
> 现网比例看 `(dfkv_ram_payload_bytes − dfkv_ram_large_used_bytes) / dfkv_ram_arena_bytes`（payload 含 dedicated 大对象，arena 不含）：
> `dfkv_ram_used_bytes − dfkv_ram_payload_bytes` 是 slot 取整的损失，
> `dfkv_ram_arena_bytes − dfkv_ram_used_bytes` 主要是 extent 尾部和回收预留的空 slot。
> 值常在 8 MiB 以上时可把 `DFKV_RAM_TIER_EXTENT_BYTES` 设为 `134217728`（128 MiB，
> 不超过 `--max-msg` 的值都放得进去），改动需重启，RAM 层内容随重启清空。代价：
> extent 数降到四分之一，而每个 shard 至少要 32 个 extent，所以 RAM 预算要
> 32 GiB × shard 数 / 8 以上才保得住 shard 数（默认 8 个 shard 时是 32 GiB）；尺寸类之间再平衡一次搬整个 extent，搬一次淘汰的数据
> 也多四倍。

#### f. 同主机/会合复本（node dedup，v2.0.0 新生产杠杆）

> 共享同一个 dedup 段（`/dev/shm` 里按 uid 和命名空间区分）的进程必须在同一个 PID 命名空间里（同一个 pod，或 `hostPID`）。判断段的创建者、取数者、发布者和 GPU 注册项是否还活着，用的都是 pid；PID 命名空间不同而共享 `/dev/shm` 的容器会把对方活着的条目当成已死或已换人而收回（dedup 时好时坏，不会返回错的数据）。段的创建者在创建中途被杀时，下一个进程等 1 秒后接手完成（创建者已死，或段还是空的）；更早版本创建、尺寸已定但没写完的段仍按原样放弃（这次进程不走 dedup）。

| env | 默认 | 说明 |
|---|---|---|
| `DFKV_CLIENT_NODE_DEDUP` | `off` | 主开关：跨 rank/process 的同主机同 key 请求会合（仅 client 侧；vLLM replicated-MLA 拓扑自动开） |
| `DFKV_CLIENT_NODE_DEDUP_GPU` | `off` | GPU 目标和 rendezvous（须 CUDA IPC）；off=host 和 rendezvous |
| `DFKV_CLIENT_NODE_DEDUP_LOG` | `off` | 会合决策 log |
| `DFKV_NODE_DEDUP_SLOTS` | `65536`（2 的幂） | dedup 表容量（power-of-2 强制） |
| `DFKV_NODE_DEDUP_GPU_SLOTS` | — | GPU dedup 表容量。另有一张固定 64 项的进程注册表（每个开启 GPU dedup 的进程占一项，与本项无关）：项里记着进程的 pid、启动时间和本次进程映像的随机标识；pid 已死，或 pid 被别的进程（含原地重启后同 pid 的新进程）占用，都会被回收。更早版本写下的项没有这些标识，只按 pid 判断，所以升级后第一次原地重启仍可能留下旧项。64 项全被活进程占满时，之后的进程打一条 INFO（`gpu-dedup: registry full`）并不走 GPU dedup |
| `DFKV_NODE_DEDUP_TTL_MS` | `5000` | dedup entry 过期（从发布时刻算起）；低于 `DFKV_NODE_DEDUP_WAIT_MS` 时启动告警：晚于 TTL 到达的 peer 会重复远端读取 |
| `DFKV_NODE_DEDUP_WAIT_MS` | `500` | follower 等待 leader 的上限。一次批量调用整批共用这一个上限（不是每个 key 各等一次；host 与 GPU 路径相同），超时的 key 合并成一次直连批量读取；leader 放弃（未命中、目标节点冷却、I/O 失败，GPU 路径还有发布前设备同步失败）时 follower 立即回退，不等满。**取值不小于最大一批读取的耗时**：leader 要等自己整批直连读取返回后才发布这批 key，耗时超过本值时，各 follower 在到期时把仍在读取的 key 合成一批自行重读，这一批不再共享（等于读了两遍） |
| `DFKV_NODE_DEDUP_TAKEOVER_MS` | — | leader 失联 follower 接管时间。也用于发布中被杀的 leader：发布（拷入 staging arena）超过这个时间的 slot 由下一次查找收回，写入或删除该 key 时立即收回（host 与 GPU 路径相同）。GPU 路径下收回 slot 的进程本身被杀时，超过这个时间后由下一次写入、删除或查找代为完成。写入或删除一个 key 时在某个 slot 上等待超过这个时间，GPU 路径会打一条 WARN（`gpu-dedup: invalidation, slot ...`）并计入 `dfkv_client_gpu_dedup_invalidate_stuck_total`，然后继续等。已知仍会卡住该 key 的情形：同机更早版本的发布者（不能收回，仍按原样等待）；当前版本进程恰好在几个只有几条指令长的窗口里被杀（锁住 slot 之后、打标记之前；提交或放弃的代际交换之后、最后一次写入之前；收回 slot 之后、打标记之前；完成收回时代际交换之后、最后一次写入之前：这一种写入和删除不再等它，只有查找仍会等到期后自行读取）。同一个值也是回收死 fetch 的门槛：fetch 状态超过这个时间的 slot，任何 key 的认领都可以收回，不必等同一个 key 再被请求（否则不再被请求的 key 留下的死 fetch 会占满 8 个探测位，落在那里的其他 key 只能不走 dedup）；被收回的慢 fetch 只是失去共享机会，不会发布错的数据 |
| `DFKV_NODE_DEDUP_ARENA_MB` / `DFKV_NODE_DEDUP_GPU_ARENA_MB` | — | dedup 复用 staging arena 大小（重点检查 GPU 复本顿） |

#### g. 一般基础设施

| env | 默认 | 说明 |
|---|---|---|
| `DFKV_LOG` | `INFO` | log level（debug/info/warn/error，大小写均可） |
| `DFKV_FANOUT_THREADS` | `32` | TCP batch、RDMA write 和兼容路径共用的 process-wide helper 上限；RDMA read 不再使用该线程池 |
| `DFKV_RDMA_READ_WORKERS` | `7` | process-wide RDMA GET/GET-Auto/SG-GET/Exist shard worker 硬上限；caller 不参与执行，并发 batch 不会继续推高活跃 read QP；batch 间按 shard round-robin。同一节点的 shard 同时最多占用 worker 数减一（默认 6），留一个 worker 给其他节点：一个慢节点占不满全部 worker，但两个慢节点同时存在时仍可能合计占满全部 worker，其他节点的读会排队。代价：环里只有一个节点的进程对它只用 worker 数减一个并发读，如有影响可调大本项。设为 1 时不预留；设为 2 时单节点上限为 1。未标明节点的 shard 不受此限 |
| `DFKV_RDMA_ENDPOINT_CACHE_MAX` | 自适应（下限 `256`） | 所有 client handle 共享的 live endpoint 上限；按 `节点数 × 2 pools × max(pool 上限, configured rails) × 1.25` 自动放大；显式设置任一预算 env 会固定全部预算 |
| `DFKV_RDMA_QP_BUDGET` | endpoint 上限 | process-wide QP 预算 |
| `DFKV_RDMA_WR_BUDGET` | endpoint 上限 × depth | process-wide negotiated WR slot 预算 |
| `DFKV_RDMA_REGISTERED_BYTES_BUDGET` | endpoint 上限 × depth × 2 × receive slot（沿用保守 admission 上限） | 实际 live endpoint 只租用 resident receive slot；GET exact grant 按操作申请，不保留连接级 pull arena |
| `DFKV_RDMA_RESOURCE_ACQUIRE_MS` | `10000` | 建立新 QP 前等待全部资源额度的上限；超时明确失败，不超发 |
| `DFKV_BATCH_CONCURRENCY` | `0`=auto（auto: `min(max(nodes, 8), 32)`） | TCP/write/兼容路径的 batch 分段并发度。RDMA read 的实际 process-wide 并发由 `DFKV_RDMA_READ_WORKERS` 限制 |
| `DFKV_GET_MISS_RETRIES` | `1` | client 批量 GET 对节点回答“不存在”的 key 的重试次数（只重试两种瞬态：节点回答“不存在”，以及节点仍健康时的 I/O 错误（连接更替；客户端分不清超时和断连，所以刚应答过的节点上超时的分片也会重发一次）；被拒、长度不符、节点已进入冷却都不重试；未命中超过一半的批次不重试）。三条批量读路径（定长、变长、SG）相同 |
| `DFKV_RETAIN_REPROBE_MS` | `30000` | 某个节点被判定不支持 retention（kRetain）之后，过这么久再问一次（节点可能已升级或被替换）；在此之前对它的 REFRESH 退化为 Exist，SET 返回 unsupported。`0` 表示每次都问 |
| `DFKV_PROBE_INTERVAL_MS` | `0`=关 | client 对 server 的主动 probe 间隔（一次 Exist 往返）。用途是逐 peer 延迟指标，以及让回避中的节点提前被试探：probe 得到应答只把该节点的回避期提前结束一次，下一个请求作为试探，由试探结果决定是否恢复；probe 本身不判定节点健康（store 已失败的节点对 Exist 仍应答未命中）。probe 促成的那次试探一旦失败，在有请求成功之前，probe 应答不能再缩短这次事故的回避期，回避按正常的翻倍节奏走完 |
| `DFKV_PEER_COOLDOWN_MS` | `2000` | PeerHealth endpoint 快速回避窗口。一次事故只算一步：回避期内到达的失败（同一批的其他分片、当时在途的请求）不再延长回避。回避期结束后只放行一个请求作为试探，其余请求继续跳过该节点（返回未命中，与回避期内相同），直到试探成功（恢复）或失败（进入下一档回避）。并发上限是每个本窗口一个试探请求，不是总共一个：试探卡在超时里超过一个本窗口，下一个请求接着试探；卡住的那个最终失败时若下一档回避已经过完，算作又一次失败。试探若对节点没有结论（请求被本进程的传输预算拒绝、没有可用 rail、或传输层因该地址刚连接失败而未拨号直接拒绝），立即把试探机会让给下一个请求，回避不加长。节点给出的任何状态（命中、未命中、cache full、quota exceeded）都算节点存活。`0` 不能关闭回避（关掉会让每个请求都去重试已死的节点）：启动时打 WARNING 并沿用默认值 |
| `DFKV_PEER_COOLDOWN_MAX_MS` | `30000` | PeerHealth 回避上限。回避时长只在试探失败时翻倍（2 s、4 s、8 s … 至本上限） |
| `DFKV_PEER_BAD_GRACE_MS` | `1000` | PeerHealth 转 bad 前的宽限期。`0` 同样不被接受：打 WARNING 并沿用默认值 |
| `DFKV_DISK_HASH_WEIGHT` | — | 盘内 Ketama 虚拟节点权重（数据中平衡关键调优，改动需重启） |
| `DFKV_DEGRADED_MIN_DISKS` | `1` | server：盘 store 终态故障后，至少还剩这么多块健康盘时节点降级运行、留在 placement（坏盘的 key 改由健康盘承接；本轮不承接的盘不算）；少于它则退出 placement。大于盘数按盘数算（任一盘故障即退出）；`0` 或非整数拒绝启动 |
| `DFKV_READ_COALESCE` | `0` | TCP 端读 convoy 合并主开关（见 README Recommended tuning 表） |
| `DFKV_READ_COALESCE_RECUR_MS` | `1000` | 复制重现窗口（copy fingerprint 64 字节） |
| `DFKV_READ_COALESCE_TIMEOUT_MS` | `500` | follower 等待 leader 上限 |
| `DFKV_READ_MAX_CONNS` | `8` | client 每节点并行 read-shard 连接上限；server 不读取 |
| `DFKV_READ_SHARD_KEYS` | `16` | client 每个 read shard 的目标 key 数；server 不读取 |
| `DFKV_MDS_IO_TIMEOUT_S` | `60` | MDS 控制面帧读写超时（MDS 进程该） |
| `DFKV_MDS_ETCD_PROBE_MS` | `30000` | MDS 与 etcd 存活探测窗（MDS 进程该） |
| `DFKV_MDS_STATS_PUT_MS` | `60000` | 仅统计（STA1）变化时重写 member 记录的最短间隔（`0`=每次心跳都重写，上限 600000）；放置相关的变化（ip、端口、权重、健康）立即写 |
| `DFKV_MDS_LIST_CACHE_MS` | `500` | member list 回答的复用时长（`0`=每次读 etcd，上限 10000）；client list 用同一个值但最多 1000（低于客户端的轮询间隔，冷却改道的登记门要求）；本 MDS 的写入会立即作废它 |
| `DFKV_MDS_FENCING` | 关 | member fencing：`dfkvctl member fence <id>` 一步把节点移出环并保持移出（无租约的 `/dfkv/v1/groups/<g>/fenced/<id>` 记录；其注册和心跳被拒，记录被删除，list 不再列出），`unfence` 恢复。改变 etcd 布局并多一次读，**同一 group 的全部 MDS 都要开启**（没开的 MDS 会让被 fence 的节点写回记录，list 仍会过滤掉它） |
| `DFKV_DRAIN` | `0` | 允许本节点排空（`dfkvctl drain`、停止前排空），需要 `--mds`；见 [§4c-1](#4c-drain) |
| `DFKV_STOP_DRAIN_MS` | `0` | `DFKV_DRAIN=1` 时，SIGTERM 后先排空并继续服务这么久再停止（`0`=不排空，上限 600000）；第二个 SIGTERM 结束等待；须小于 `terminationGracePeriodSeconds` 减去停止时间 |
| `DFKV_MDS_UNWRITTEN_HOLD_MS` | `120000` | MDS 把心跳已到但没写进 etcd 的 member 视为存活的时长；期间 etcd 缺少该 member 则 member list 返回错误（`0`=关，上限 600000；须大于 MDS 端点数 × 10s 心跳间隔；见 §2b） |
| `DFKV_RDMA_HEALTH_RECOVERY_SAMPLES` | `3` | 仅 RDMA listener 生效。任一 initialized rail 健康即可 placement；部分掉轨保持 `PARTIAL`。最后一条健康 rail 丢失立即 `DEGRADED`；0→非 0 恢复要求连续成功样本，通过后 `PARTIAL`/`ACTIVE` 回环 |
| `DFKV_RDMA_HEALTH_FILE` | 未设置（生产必须未设置） | 仅受控故障注入；逐 initialized device 提供 `device port_state phys_state`，生产从 sysfs 读取 |
| `DFKV_TENANT_QUOTAS_COUNT` | — | tenant 配额文件预期条数（容量校验） |

#### 未收录的常见误配 → 详见 CONNECTORS.md §1.7

客户端设置于 dfkv_server 不生效的 env 包括（不是服务端配置，**将此会静默无效**）：
`DFKV_RDMA_DEV`（server 读 `--rdma-dev`）、`DFKV_NODE_DEDUP_*`（全部 client 侧）、
`DFKV_ACCESS_LOG_*`、`DFKV_CLIENT_*`、`DFKV_METRICS_*`、`DFKV_TRACING_*`、
`DFKV_LIB`/`DFKV_BUILD`（客户端 lib 路径）等。

## 4. 集群成员管理

### 4a. MDS 动态发现（推荐）

节点通过 `--mds` + `--group` + `--id` + `--advertise` 向 MDS 自注册。客户端在
`dfkv_open_v2` 构造时一次性提供 endpoints/group/poll interval。现代客户端请求
`kListTopology`：placement 仍用原成员字段和 `MembersEpoch` 独立去重；已有
`HLT1` 扩展承载 eligibility 与逐设备健康，`MembersTopologyEpoch` 变化时即使
Ketama placement 不变也调用 transport 的 peer-topology 更新。当前 MDS 中的旧
member 因缺少 HLT1 而产生 incomplete topology：未配置 tiers 时 homogeneous
兼容，配置 `DFKV_RDMA_RAIL_TIERS` 时 fail closed。旧 MDS 不支持
`kListTopology`，modern poll 失败并保留 last-good ring；rollout 必须 MDS-first。
两层离线检测：
- **层 2（权威）**：etcd lease 到期（TTL 30s）→ 下次 MDS poll（默认 3s）→
  placement-content epoch 推进 → 环重建。小幅缩容要**同一值连续 2 次 poll**
  才采纳（通常约 TTL+两次 poll）：停机恢复时成员逐个重注册、每个视图都在
  阈值以内的斜坡因此不会逐个视图重建环；小幅扩容立即采纳。代价：一直在变的
  视图在变化期间不会被采纳。每次 poll 少一个成员的持续下降要等它停下来才采纳；
  在 N 和 N−1 之间来回抖动的成员一直不会被移出环。客户端仍会经 `PeerHealth`
  的 cooldown 立即跳过已死的节点，所以代价是这些 key 的 miss，不是数据放错；
  偏离 `DFKV_MDS_SHRINK_GUARD_PCT`（默认 50，0=同时关闭 shrink/growth
  两臂，空视图臂常在）的批量视图要走 adoption guard：比较基准是**冻结的
  trusted reference**（不随每跳下移，挡住 diffuse mass-expiry 逐跳蚕食），
  shrink 与对称 growth（停机恢复期的批量重注册）两臂可疑视图都必须以
  **同一值连续 3 次 poll** 才采纳；仍在逐跳变化的缩容/恢复斜坡每次 poll
  重置 streak，到 plateau 才一次采纳、中途不重建环。批量缩容被 guard
  挂起时默认最坏约 30s+3×3s，而不是固定“≤30s”。
  客户端已采信空环之后，回来的成员视图边来边采纳（总比没有环好），
  trusted reference 要等某个值连续 3 次 poll 才锚定，不会把最先回来的几个
  节点当成整个环冻结住；锚定之前，缩容一侧以上一次采纳的视图为临时基准，
  斜坡中途一次远低于它的坏视图仍会被挡住。
  etcd 拒绝写入造成的 lease 批量过期不依赖这道 guard：MDS 直接拒绝下发残缺
  视图（list 返回错误，客户端保留 last-good ring 与 peer topology），见 §2b。
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

`systemctl start`/`stop` 都带 `--no-block` 下发：systemd 只把作业排进队列就返回，脚本
不等作业结束。`--command-timeout` 只约束单条 `dfkvctl`/SSH 命令，这些命令都很快返回。

- **成功不等旧 unit 停完。** 停止中的 `dfkv_server` 先停心跳，旧节点的注册在租约到期
  （约 30 秒）后就从环里消失，而 unit 可能还要刷几分钟积压
  （`DFKV_RAM_SHUTDOWN_DRAIN_TIMEOUT_S`，最坏到 `TimeoutStopSec`）。旧节点注册消失、
  新节点在放置环内且就绪、client 没少，脚本即成功退出，并在结束事件里注明旧 unit 可能
  还在停止；它会自己停完，不用再做什么。所以 `--timeout`（每个状态迁移的上限，默认
  180 秒）不需要覆盖 unit 的停止时间。
- **回滚要等旧 unit 停完再起来。** 停旧之后失败，脚本下发 `systemctl start --no-block`；
  unit 还在停止时，systemd 把这个启动排在停止之后，停完才执行。脚本先等 unit 变回
  `active`（上限为 `--stop-timeout` 加 `--timeout`），再等它重新入环并就绪——在这之前，
  环里可能还留着正在退出的那个进程的注册。`--stop-timeout` 默认 660 秒，要大于 unit 的
  `TimeoutStopSec`（示例 unit 是 600）。超过上限时以 4 退出，报错会说明 systemd 已经
  接受了这次启动、停完后会执行，需要人工确认节点回环；只有启动命令本身没发出去时才要求
  手工启动。
  **以 4 退出且报错是 “rollback not confirmed … systemd accepted the start” 时，等旧 unit
  稳定下来再重跑脚本**（`systemctl show --property=ActiveState dfkv` 不再是
  `deactivating`/`activating`，并看清旧节点有没有回到环里）。这段时间旧节点还没注册，
  重跑会按“旧节点已退出、新节点健康”打印 nothing to do 并以 0 退出，而排着队的启动之后
  仍会把旧节点带回环里。
- **新节点已启动、旧节点未停时失败：** 脚本停掉新节点并等它的 unit 停完（上限
  `--stop-timeout`），它的注册随租约过期。
- **就绪只看被问的那个节点。** `dfkvctl stat --all` 只要组内有成员不可达、不健康或健康
  未知就以 1 退出，但表格照常打印。脚本读该节点自己那一行（可达，`READY`、`HEALTH` 为
  `yes`，`MDS`、`RDMA` 不为 `NO`），别的成员——包括正在被替换的那个——不影响判断。没有
  表格的非零退出（连不上 MDS、空环）仍按命令失败处理。
- **新节点要在放置环内（eligible），旧节点只需已注册。** 旧节点可以已经被 MDS 移出
  放置环（例如 IB 轨全断），这正是常见的替换原因；这时停它不改变放置，回滚也只要求它
  重新注册。

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

前提：目标机已安装并配置同名 `dfkv.service`，执行机可 BatchMode SSH，执行机上的
`dfkvctl` 支持 `ring --json`（与脚本同一发行版）；新旧节点 ID
必须不同（同 ID 会争抢 etcd lease，脚本拒绝）。`--timeout` 是**每个状态迁移**的
上限而不是无限等待；`--min-clients 0` 允许还未升级客户端注册能力的旧车队，但脚本
仍会调用 `dfkvctl clients`，并保证执行前已观察到的 client 在切换后没有消失。

### 4c-1. 排空（drain，`DFKV_DRAIN`，默认关）<a name="4c-drain"></a>

排空中的节点保留它在环上的位置、照常服务读，2.43 起的客户端不再往它写新值：一个
key 的写入改去“没有排空节点的环”上的 owner（排空节点离开后它本来就归那里），读先
问那个新 owner，未命中且 key 原 owner 是排空节点时再问排空节点。REMOVE 和 RETAIN
同时发给两边：REMOVE 两边都确认才算成功，否则排空节点上的旧值会被下一次读读回来；
RETAIN 保护两份（排空节点那份还在服务读），哪边有 key 就用哪边的结果。排空不搬数据、
不改放置（`placement_epoch` 不变），节点离开时期间写入的值已经在它们最终的位置上。
环上全部可用节点都在排空时没有别处可写，客户端照旧写。

- **开关与升级顺序。** 节点设 `DFKV_DRAIN=1`（需要 `--mds`）才会报告排空状态、接受
  `dfkvctl drain`。状态作为成员记录的可选扩展 DRN1 下发，只在有节点排空时才写，没有
  排空节点时 MDS 和节点发出的字节与之前完全相同。MDS 须先升级到 2.43：旧 MDS 会丢掉
  这个扩展。**同一 group 的全部 MDS 都运行 2.43 或更高之前不要排空**：MDS 新旧混用时，
  registrar 在多个 MDS 间切换，落到 2.42 MDS 上的每次心跳都会把记录重写成不带 DRN1
  的样子，排空状态在两次轮询之间时开时关；客户端的读写随之在两个 owner 间来回切换，
  在“关”的间隙里发出的 REMOVE 只到达排空节点，已删除的 key 之后可能又被读到。
  2.42 及之前的客户端忽略它，照旧读写排空节点，节点也照常服务它们。
- **`dfkvctl drain <ip:tcp端口>`** 让节点进入排空，`--undrain` 退出；节点在 MDS 写入新
  记录后才回答（最多等 15 秒），客户端在下一次轮询（默认 3 秒）内生效。节点没开
  `DFKV_DRAIN`、没有 MDS 或版本早于 2.43 时回答“does not drain”；节点已被 fence
  （`dfkvctl member fence`）时 MDS 拒绝它的心跳，`dfkvctl drain` 报的是“its MDS has not
  taken the record yet”那一种（节点本地已进入排空，但 MDS 里没有它的记录）。`dfkvctl ring`
  的 STATUS 列显示 `DRAINING`；`ring --json` 的 `status` 仍是轨健康，另有
  `"draining": true`。和 kRemove 一样，kDrain 没有额外鉴权，依赖同样的网络信任边界。
- **被 `dfkvctl drain` 排空后停止 = 下线。** 停止时节点向 MDS 删除自己的记录
  （kDeregister），客户端一次轮询内把它移出环，不用等 30 秒租约。处理删除的 MDS
  立即作废自己的列表缓存；同组其他 MDS 实例最多再用缓存回答
  `DFKV_MDS_LIST_CACHE_MS`（默认 500 ms），按默认值仍在一次轮询（3 秒）之内。重启时新进程不带
  排空状态。
- **停止前排空：`DFKV_STOP_DRAIN_MS`**（默认 `0`=不排空，上限 600000）。收到
  SIGTERM 后节点先进入排空、继续服务这么久，再走原来的停止流程；第二个 SIGTERM 立即
  结束等待。这种排空不删记录（多用于重启：新进程接回同一位置）。
  **Kubernetes：`terminationGracePeriodSeconds` 必须大于 `DFKV_STOP_DRAIN_MS`
  加正常停止时间**（RAM tier 刷盘最长 `DFKV_RAM_SHUTDOWN_DRAIN_TIMEOUT_S`），否则
  kubelet 在排空中途 SIGKILL；systemd unit 的 `TimeoutStopSec` 同理。
- **混用新旧客户端的已知限制：** 旧客户端仍把排空节点的 key 发给排空节点。新客户端
  排空期间写到新 owner 的 key，被旧客户端 REMOVE 时只删到排空节点，新 owner 上那份
  在被淘汰或被新客户端删除前仍可读到。反过来没有问题：旧客户端写到排空节点的值，新
  客户端未命中后会去读。**依赖 REMOVE 的部署（LMCache L2 淘汰、vLLM 清理旧布局对象）
  先把客户端全部升级到 2.43 再排空。**
- 指标：客户端 `dfkv_client_ring_draining_members`、
  `dfkv_client_drain_requests_total{op,result}`；MDS `dfkv_mds_deregister_requests_total`。

### 4c-2. 消失成员宽限（absent grace，`DFKV_RING_ABSENT_GRACE_MS`，默认 0=关）<a name="4c-absent"></a>

客户端开关，只对 MDS 动态发现生效（静态成员表没有成员会消失）。设计见
[ring-unavailable-owner.md](design/ring-unavailable-owner.md) §3.1。

- **作用：** 一个成员从 MDS 视图里消失（租约过期、记录被删、轨健康不合格），客户端不立即把它
  移出环，而是在宽限期内把它当作“缺席”：不再写它，REMOVE 和 RETAIN 除了发给新 owner 也发给
  它。宽限期满（客户端每秒检查一次）才移出；宽限期内回来的成员立刻恢复为普通成员。
- **不读缺席成员：** 新 owner 未命中后不去问缺席成员。成员不在视图里时启动、从没见过它的
  客户端，把 REMOVE 只发给新 owner，缺席成员上可能留着已删除的副本；读它属于回读
  （设计 §3.4，后续阶段），要按回读的规则来。
- **它带来什么：** 成员带着数据回来时，删除是一致的。测试环（8 个节点进程，客户端持续删除
  key）里一个成员被杀 90 秒后带着数据回来：关闭时回来后读到 4 个已删除的 key，打开时 0 个。
  命中率不变：删除一个成员的记录 40 秒，开关两边都是最差一秒 0%、重算 27 MiB。
- **上限：** 宽限期最多 600000 ms（更大的值按 600000 处理并打 WARNING）；同时缺席的成员
  最多 `max(1, N/8)` 个（N 为消失前环里的成员数），再消失的成员立即移出。
- **计划下线不保留：** 最后一条记录是排空状态（DRN1，`dfkvctl drain` 之后下线）的成员
  立即移出。
- **成员视图保护（#497）：** 只作用于被接受的视图；被拒的视图什么都不改。
- **REMOVE：** 发给写 owner 和缺席成员。缺席成员通常已经停了，不需要它确认：REMOVE 以写
  owner 的确认为准，缺席成员没确认的部分记入队列。它回到视图时立即重发（它在缺席期间的失败
  让 PeerHealth 进入了回避期；回到视图等同一次 probe 应答，在每个客户端都提前结束一次回避期，
  没有待重发 REMOVE 的读客户端也一样，回来后的下一个请求（这里就是重发）就是那次试探），仍未确认的每秒再发（按 PeerHealth 当时的状态），直到确认，
  最多到回来后一个宽限期；同一个 key 在队列里只记一次。队列总长上限
  `DFKV_RING_ABSENT_REMOVE_QUEUE`（默认 4096）；超出的、成员被移出时还在队列里的、回来后
  一个宽限期仍未确认的都计入 `dfkv_client_ring_absent_removes_dropped_total`。
- **RETAIN：** 缺席成员上的副本也被保护（留到它回来）；它的回答不代表这个 key 现在可读，
  不计入结果。
- **与排空同时：** 某个成员在排空、另一个成员缺席时，一个 key 可能同时有写 owner、排空节点和
  缺席成员三个 owner；REMOVE、RETAIN 分别发给各自不同的那几个。排空节点必须确认 REMOVE，
  缺席成员不必。
- **已知边界：**
  - 成员带着数据回来时，仍会提供它没确认删除的 key（本客户端重发前的竞争、被队列上限丢掉的，
    以及其他客户端从未发给它的）。这与关闭时相同：今天一个离开视图的成员收不到任何 REMOVE，
    回来时同样提供这些副本。
  - 成员回来后、某次重发之前写到它上面的 PUT，可能被这次重发删掉：代价是一次未命中，不会
    读到旧值。
- 指标：`dfkv_client_ring_absent_members`、`dfkv_client_ring_absent_dropped_total{reason}`、
  `dfkv_client_ring_absent_removes_{unconfirmed,retried,dropped}_total`；对缺席成员的 REMOVE、
  RETAIN 计入 `dfkv_client_drain_requests_total`。

### 4c-3. 冷却改道（cooldown reroute，`DFKV_RING_COOLDOWN_REROUTE`，默认 0=关）<a name="4c-reroute"></a>

客户端开关，只对 MDS 动态发现生效。设计见
[ring-unavailable-owner.md](design/ring-unavailable-owner.md) §3.2。

- **作用：** 本客户端的 PeerHealth 让某个 key 的 owner 处于回避期（它刚连不上）时，这个 key 的
  PUT、GET、EXIST、RETAIN 改走一个替补节点：在本客户端近期所有环（下面）合起来的顺时针顺序里，
  前 `DFKV_RING_REMOVE_FANOUT`（默认 3）个成员中第一个“稳定”（在近期每个环里都有）且不在回避期的。
  前几个都不能用时照旧（跳过 owner：读未命中、写失败），计入
  `dfkv_client_cooldown_reroutes_total{op="none"}`。替补上未命中就是未命中，不去问 owner。
  回避期满（不触发试探的检查）就回到 owner，第一个请求照常当试探。
- **近期环：** 客户端在最近 S 里采用过或轮询看到过（不管 #497 的保护是否放行）的每个视图的写环。
  S = 11 × `poll_max`，`poll_max` 是本客户端和登记表里各客户端轮询间隔中最大的那个。
  有更慢的客户端登记、S 变大时，改道先停，等环的历史够新的 S 再开（`reason="s_growing"`）；
  S 变小时旧值再用一个旧 S。
- **REMOVE 扇出：** 开关打开时，REMOVE 发给 key 在每个近期环上的后继，直到第
  `DFKV_RING_REMOVE_FANOUT` 个稳定成员（含），再加排空节点和缺席成员，每个只发一次。
  **只有每个目标都在这次调用里确认才算成功**：哪个目标在本客户端回避期内，或失败了，REMOVE 就
  返回失败（`dfkv_client_remove_unconfirmed_total{reason="cooling"|"failed"}`）。回避期内那个目标
  的那份记入清理队列，回避期满后从客户端每秒一次的检查里补发，只为清理，不作为成功的依据。
  缺席成员照旧走 4c-2 的队列。**代价：** 一个节点在回避期内时，约 (`DFKV_RING_REMOVE_FANOUT`+1)/N
  的 key 的 REMOVE 失败（关闭时是 1/N），持续回避期那么久；调用方看到失败可以重试。要降低这个比例就把
  `DFKV_RING_REMOVE_FANOUT` 调小（能绕过的回避节点也随之变少）。一个离开视图的成员在 S 内仍是
  REMOVE 目标；它若是死了才离开的，它附近的 key 的 REMOVE 在每次删除之后的 S 内都失败。这是正确性
  所需：别的客户端的近期环里可能还有它，拿它当了替补。
- **一次回避期结束时清理替补：** 客户端记下自己写到替补上的 key（PUT 和 RETAIN SET，按 owner 和
  回避期：试探失败开始的新回避期是新的一次），记着的和等待清理的合计上限 `DFKV_RING_REROUTE_KEYS`
  （默认 65536），满了以后改道的写被拒绝（未命中，计入 `dfkv_client_reroute_refused_total`），不会有
  不记账的替补写入。一次回避期结束（owner 回避期满、开始了新的一次，或登记门关上）时，客户端把那次的
  key 从替补上 REMOVE 掉；没确认的留在清理队列里每秒重发，除了关闭时从不放弃，确认之前改道的读不读
  那一份。扇出 REMOVE 里回避期内目标的那份另有一个同样大小的上限，超出的放弃并计数。正常关闭
  （`dfkv_close`，重启也走它）先做同样的清理，最多 `DFKV_RING_REROUTE_CLOSE_MS`（默认 2000，0=不清理），
  没发出去的计入 `dfkv_client_reroute_cleanup_total{result="abandoned"}` 并打日志。开关关闭时 close 不变。
- **登记门：** 只有 MDS 客户端登记表里本 group 的每个客户端都带着扇出标记时改道才生效。本版本的库在
  开关打开时往自己的登记信息里加 `dfkv-remove-fanout=1;poll_ms=<轮询间隔>`。登记表里有不带标记的
  客户端（旧版本，包括只读的连接器，或者没开这个开关的）、标记里没有 `poll_ms`、登记表的回答超过两个
  轮询间隔没更新、本客户端没有开 MDS 发现或没有登记，或者轮询间隔小于 1000 ms，改道都关闭（扇出仍然
  打开）。门的状态看 `dfkv_client_registry_gate_open`，没带标记的客户端数看
  `dfkv_client_registry_unmarked_clients`，每次由开到关按原因计入
  `dfkv_client_registry_gate_failures_total{reason}`。
- **上线方式：** 要用改道，group 里每个登记的客户端都要打开本开关（读客户端也要：门分不清谁会删除）。
  打开了本开关、但没有开 MDS 发现就登记的客户端不带标记（标记里的轮询间隔来自发现），算作没带标记，
  让所有客户端的门都关着（安全的一侧）；C API 总是先开发现再登记。
  不登记的客户端看不见，不在保护范围内：只在所有会删除的客户端都登记的 group 里打开。一个本版本、
  开关关闭的客户端打开后马上 REMOVE，在门看到它的登记之前（约一次登记加一次轮询加 MDS list 缓存的
  时长）它的 REMOVE 只到 owner，和今天任何客户端一样。
- **已知边界：** 一个只存在不到一个轮询间隔的视图，若是唯一给出替补 T 的视图，扇出可能漏掉 T。
  正常关闭前被杀掉（或关闭超时）的客户端，它写到替补上的副本留到替补把它们淘汰为止。
- **代价：** 每个 REMOVE 最多 `DFKV_RING_REMOVE_FANOUT`+2 次往返（并行），视图变化时更多；每个写到替补的
  key 回避期结束时多一次 REMOVE；客户端每次轮询多一次 `kListClients`（MDS 用 list 缓存回答）。
- 指标：上面各项，加 `dfkv_client_remove_fanout_total{result}`、
  `dfkv_client_cooldown_reroutes_total{op}`、`dfkv_client_reroute_cleanup_total{result}`。

### 4c-4. 近期删除记忆（recent-REMOVE memory，`DFKV_REMOVE_MEMORY_MS`，节点侧，默认 0=关）<a name="4c-remove-memory"></a>

节点开关。设计见 [ring-unavailable-owner.md](design/ring-unavailable-owner.md) §3.3。它是回读（§3.4，
后续阶段）的前提：回读之前先问这些节点“这个 key 最近在你这里删过吗”。单独打开它不改变任何读、写、
删除的结果。

- **记什么：** 本节点确认过的每个 REMOVE（不管 key 当时在不在），在回复之前记下 key（含租户）和时间，
  只在内存里。之后对同一个 key 的 PUT 照常存、照常读，不清除这条记录。
- **多久、多大：** 记录保留 `DFKV_REMOVE_MEMORY_MS`（打开时常用 600000，上限 3600000）；最多
  `DFKV_REMOVE_MEMORY_ENTRIES` 条（默认 1048576，每条 40 字节，满时 40 MiB；不超过 4194304 条时按需翻倍增长，每次翻倍让 REMOVE 停顿几毫秒，超过时在启动时一次分配全部内存（条数 × 32 字节，加索引：4 字节 × 不小于 条数/0.7 的 2 的幂），之后不再增长，范围
  1024–67108864），满了丢最旧的，
  节点“知道”的时间段随之变短。回读要求它至少覆盖客户端的回读窗口加十个最长的轮询间隔（§3.4）。
- **重启：** 只在内存里。节点启动（包括带数据的热重启）之后只知道启动以来的删除，问更早的时间段
  答“不知道”，回读据此不回读。
- **协议：** 新增 op `kRemovedWithin`（25），TCP 和 RDMA 都支持。关闭时（以及旧版本节点）对它回
  `kInvalid`，和任何未知 op 一样；客户端把它当作每个 key 都“不知道”，不记入 PeerHealth，60 秒内不再问
  这个节点。旧客户端从不发这个 op；已有 op 和格式都不变。
- 指标：`dfkv_remove_memory_entries`、`dfkv_remove_memory_covered_seconds`、
  `dfkv_remove_memory_dropped_total`、`dfkv_removed_within_queries_total{result}`（只在打开时输出）。

### 4c-5. 回读前任 owner（read-back，`DFKV_RING_READBACK_MS`，客户端，默认 0=关）<a name="4c-readback"></a>

客户端开关，只对 MDS 动态发现生效。设计见 [ring-unavailable-owner.md](design/ring-unavailable-owner.md) §3.4。
要求节点打开近期删除记忆（4c-4），且 `DFKV_REMOVE_MEMORY_MS` 至少为本窗口加十个最长的轮询间隔。

- **作用：** 环的写入成员变了（成员加入或离开、开始或结束排空/缺席，或者一次冷却改道结束）之后的
  窗口内，一个 key 的现 owner 回答“不存在”时，客户端先问这个 key 的一组节点“最近在你这里删过吗”，
  全都回答“没删过”才去它的前任 owner 读一次。新 owner 是空的（换了新节点、带空盘回来）时，
  前任 owner 上的值还能用上。
- **问谁：** 本客户端近期每个环和窗口里保留的旧环上，这个 key 的前 `DFKV_RING_REMOVE_FANOUT` 个后继，
  加排空节点和缺席成员，**不问现 owner**（它刚回答“不存在”；刚启动的它也什么都记不住）。任何一个
  回答“删过”或“不知道”（没有记忆、记忆不够长、问不到、整个请求失败）都不回读。
- **前提：** `DFKV_RING_REMOVE_FANOUT` 至少 3，否则不启用并打 WARNING；登记门打开（与 4c-3 相同：
  group 里每个登记的客户端都要打开 4c-3 或本开关）；窗口之内只有一次变化（第二次变化把窗口关掉）；
  保护视图（#497）把这次变化压了超过九个轮询间隔的，这次变化没有窗口。
- **与冷却改道（4c-3）一起用：** 打开本开关的客户端在登记信息里多带 `readback=1`。登记的客户端
  **全部**带 `readback=1` 时，一次冷却结束后写到替补上的 key 留到窗口结束才清理（这期间只经回读读到，
  前任 owner 带空盘回来时还能用上）；有一个不带，就像 4c-3 一样在冷却结束时立刻清理。这个条件每个 tick
  和每次登记应答都重新检查：一旦有客户端不带 `readback=1`（新登记的或去掉了的）、应答过期或登记门关上，
  所有等待中的 key 立刻清理，不再等待。新登记的不带 `readback=1` 的客户端最多一个轮询加列表缓存 TTL（≤ 1 s）
  加一个 tick 后才被看到；本版本只开 4c-3 不开本开关的客户端在看到任何 `readback=1` 时，读替补上不是自己
  这次冷却写的 key 一律按未命中处理，所以这段时间里只有更老的版本可能读到它们。打开本开关的
  客户端读替补上不是自己这次冷却写的 key 时（GET、EXIST，以及 RETAIN 的 REFRESH 和 SET，它们的回答也报告
  key 在不在），也先问记忆（同回读），全都“没删过”才读。
  `dfkv_close` 立刻清理等待中的 key；进程被杀则留到替补淘汰它们（其他客户端读它们仍先问记忆）。
- **`DFKV_RING_REROUTE_KEYS` 的大小：** 等待窗口的 key 也算在这个上限里。打开本开关后，它要容纳
  约“改道写入速率 ×（一次冷却 + 本窗口）”个 key；满了以后改道写入被拒（调用方得到未命中，
  计入 `dfkv_client_reroute_refused_total`），不会出错，只是少存。例如每秒改道写 500 个 key、冷却 30 s、
  窗口 60 s，需要约 45000（默认 65536 够用）。
- **代价：** 每个回读候选一次批量记忆查询（通常 3 个节点）加一次读；现 owner 命中的读没有任何额外开销。
  窗口最长 `600000 − 10 × 轮询间隔` ms（更大的值按此处理并打 WARNING）。
- 指标：`dfkv_client_readback_reads_total{result}`（hit、miss、refused_removed、refused_unknown、
  refused_length、refused_gate、refused_cooling）、`dfkv_client_readback_changes_refused_total{reason}`
  （held、chained）、`dfkv_client_readback_window_active`、`dfkv_client_readback_memory_short_total`
  （节点的记忆长期短于需要的时长，每个节点一条 WARNING）。

### 4c-6. 回读重写过的 key（rewrite，`DFKV_WRITE_MEMORY_ENTRIES` 节点侧 + `DFKV_RING_READBACK_REWRITE` 客户端侧，默认都关）<a name="4c-rewrite"></a>

设计见 [ring-unavailable-owner.md](design/ring-unavailable-owner.md) §3.5（CLI-35）。

- **作用：** 回读（4c-5）遇到“某个节点说最近删过”的 key 一律不回读。打开后，如果前任 owner 在那次删除之后
  又收到过这个 key 的 PUT（删了又写回来），就照样回读。删除和重写交替频繁的负载（例如读到缺失就重新计算写回）
  在节点空盘回来后的头几秒命中率更高。
- **怎么打开：** 节点设 `DFKV_WRITE_MEMORY_ENTRIES`（最近 PUT 记忆的条数，0=关，1024–67108864；
  需要同时打开 4c-4 的 `DFKV_REMOVE_MEMORY_MS`，窗口与它相同），客户端设 `DFKV_RING_READBACK_REWRITE=1`
  （需要 4c-5 的回读）。客户端开关关着时什么新请求都不发；混用老版本节点（不认识新请求）或没开记忆的节点时，
  自动退回 4c-5 的规则（照旧不回读）。
- **只在这些机器上打开：** 物理机，或者虚拟机暂停/迁移后 guest 的时钟会追上实际时间的虚拟化平台。判断依据：
  节点比较的是各自 `CLOCK_MONOTONIC_RAW` 测出的“多久以前”，要求这个时钟不停走；节点每秒检查
  `CLOCK_REALTIME`、`CLOCK_BOOTTIME` 相对它的漂移，窗口内累计超过 1 秒就清空记忆（只是少回读），但 guest
  时钟在暂停期间停住、要等 NTP 之后才纠正的平台查不出来，**不确定的话不要打开**。
- **大小：** 每条 40 字节（与 4c-4 同样的结构，分 16 段）。需要容纳“窗口内的 PUT 数”：每秒 PUT 数 ×
  `DFKV_REMOVE_MEMORY_MS`/1000。例如每秒 2000 个 PUT、窗口 600 s，需要约 120 万条（约 48 MiB）；放不下时只记得
  更近的 PUT，少回读，不会多回读。超过 4194304 条时内存在启动时一次分配，之前按需翻倍（每次翻倍让这一段的 PUT
  停顿几毫秒）。每个确认的 PUT 多一次约 0.3 µs 的记录（段锁内）。
- **判断规则：** 删除的年龄、写入的年龄都由各节点自己报告，客户端用自己发请求、收回复的时间夹住它们，按时钟
  速率误差 1000 ppm 和 1 秒的暂停余量比较，所以删除之后约 1.2 秒以内的重写不算。同 key 的完成与查询按同一把 key 锁排序，同毫秒也不会把先写后删误认成后写。
- 指标：节点 `dfkv_write_memory_entries`、`dfkv_write_memory_dropped_total`、
  `dfkv_write_memory_cleared_total{reason}`（realtime、boottime：时钟漂移超过 1 秒而清空）；客户端
  `dfkv_client_readback_reads_total{result="hit_rewritten"}`（因为重写而回读的次数，另外也计入 hit 或 miss）。

### 4d. 成员漂移 / split-brain 巡检

`deploy/dfkv_membership_audit.py` 分别查询每个 MDS（不是把 endpoints 当故障转移
列表只取一个），并从 etcd v3 gateway 只读 `/dfkv/v1/groups/<g>/members/`：
比较 placement view、节点注册值中的自报 INFO、etcd registration `mod_revision`
和 `dfkvctl ring --json` 给出的 placement epoch（MDS 与客户端比较的同一个值，
需要与脚本同一发行版的 `dfkvctl`）。任一 MDS 不可达、视图分裂、MDS/etcd
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

1. 对最终显式 rail 列表做**只读、无状态变更**的 clean preflight；设备必须存在且
   verbs open/query 成功，`DOWN`/`LinkDown` 只记录、不在这里删出配置或强制拉链：
   ```bash
   set -eu
   for dev in mlx5_0 mlx5_1; do  # 本机全部数据面 rail
     test -r "/sys/class/infiniband/$dev/ports/1/state" &&
       test -r "/sys/class/infiniband/$dev/ports/1/phys_state" || exit 1
     ibv_devinfo -d "$dev" >/dev/null
     printf '%s state=%s phys_state=%s\n' "$dev" \
       "$(cat "/sys/class/infiniband/$dev/ports/1/state")" \
       "$(cat "/sys/class/infiniband/$dev/ports/1/phys_state")"
   done
   ```
   此检查不能代替真实 anchor/MR 初始化；确认 unit 有 `LimitMEMLOCK=infinity`
   后由 `dfkv_server` 启动结果作为最终门。不得为了“先起进程”临时缩短
   `--rdma-dev`，否则固定 topology/容量语义已改变。
2. （MDS 路径）起 etcd，再起所有 `dfkv_mds` 副本（§2b），确认日志 "listening"。
3. 所有节点起 `dfkv_server`（§3）。启动日志
   `rdma topology startup: configured=N initialized=N ACTIVE=M inactive=...`
   必须显示显式部署 `configured == initialized`；允许 `ACTIVE < configured`，
   但此时 `dfkv_server_ring_eligible=0`，节点是已初始化、可监控但不承载
   placement 的 `DEGRADED` 成员。missing/open/port/GID query 或任一 anchor/MR
   失败必须使服务启动失败。
4. 冒烟（任一能访问内网的机器；只对 ring-eligible 节点执行）：
   ```bash
   dfkv_smoke --members n57=192.168.1.57:28000 --size 2752512                          # TCP
   DFKV_RDMA=1 DFKV_RDMA_DEV=mlx5_0 dfkv_smoke --members n57=192.168.1.57:28001 --size 2752512
   ```
5. 端到端零拷贝校验（插件 → libdfkv → RDMA → server，验证 payload 直落缓冲）：
   ```bash
   DFKV_RDMA=1 DFKV_RDMA_DEV=mlx5_0 DFKV_MEMBERS='n=192.168.1.57:28001' \
     python3 test/python/rdma_e2e_validate.py    # 期望 RESULT: ZERO-COPY RDMA E2E OK
   # 随后在 server /metrics 确认 v2_ready=1、v2_conns_opened_total 增长且 free_bytes 有余量
   ```
6. 压测（可选）：`DFKV_RDMA=1 DFKV_RDMA_DEV=mlx5_0 dfkv_bench --members ... --size 2752512 --count 8000 --threads 64`。
7. 在**一个受控 SGLang 副本**上切 `dynamic` 后端，发共享长前缀请求看命中上涨，确认后推广。

### 5b. 候选版本负载回归门
`DFKV_BENCH_STALL_MS=<毫秒>` 只由 `dfkv_bench` 读取。PUT 或 GET batch 超过
阈值时，会带 wall-clock 时间戳和 `phase=PUT|GET` 写 stderr；耗时使用单调时钟。
未开启时不增加逐操作 wall-clock 采样。该参数不控制 `dfkv_server`。

`--put-arena 0|1` 选择是否预注册共享 PUT 源缓冲区（默认 0）。
对照必须使用相同模式；注册失败返回非零，不静默退回逐次注册。
`DIAG_CONN` 报告每阶段 probe 与 endpoint cache 计数；失败 run 不作为成功吞吐样本。


`deploy/dfkv_load_regression.py` 对 baseline/candidate 使用完全相同的 size/count/
threads/batch/transport 环境，先 warmup，再各跑多轮 `dfkv_bench --op both`。吞吐对
trial 汇总 median/p95/p99；延迟从 workload 前后的 server
`dfkv_op_latency_seconds` histogram **差值**计算 median/p95/p99（不是从终身累计
值猜测）；错误率来自 bench `fails/count`。因此两套目标都必须开启 metrics port，
并在隔离压测窗口执行，避免其它流量混入 histogram delta。

```bash
DFKV_RDMA=1 DFKV_RDMA_DEV=mlx5_0 DFKV_RDMA_DEPTH=4 \
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
