# dfkv 推理引擎对接指南 — SGLang HiCache · vLLM 直连 · LMCache（含客户端配置参考）

> 本文是 dfkv **客户端侧**的唯一完整文档：三条对接路径（SGLang HiCache 插件、vLLM
> `DfkvStoreConnector` 直连、LMCache connector）的部署配置 + 设计实现 + 跨连接器通用的
> 客户端 env/config 总表。它合并并取代原来的 `docs/hicache/`、`docs/vllm/`、
> `docs/lmcache/` 三个目录与 `docs/CLIENT_CONFIG.md`。
>
> **前提：先按 [DEPLOY.md](DEPLOY.md) 部署好 dfkv 集群（server + MDS）**；集群侧
> flag（`--dir`/`--port`/`--rdma-port`/`--mds`/`--group`/`--advertise`…）都在那里，
> 本文只讲客户端。服务端架构（wire 协议、slab 引擎、RAM 热层）见 [ARCHITECTURE.md](ARCHITECTURE.md)。

---

## 0. 三条路径总览

| | **SGLang HiCache**（§2） | **vLLM 直连 DfkvStoreConnector**（§3） | **LMCache connector**（§4） |
|---|---|---|---|
| 上层框架 | SGLang HiCache（L3 external KV） | vLLM `KVConnectorBase_V1`（绕开 LMCache，占 Mooncake 同槽位） | LMCache `RemoteConnector`（in-process）或 MP-server `L2AdapterInterface` |
| 源码 | `integration/hicache/dfkv_hicache.py` | `integration/vllm/`（包 `dfkv_vllm`） | `integration/lmcache/`（包 `dfkv_connector`） |
| 接口 | `HiCacheStorage`（`batch_set_v1/get_v1`…） | connector API（scheduler + worker 两侧） | `RemoteConnector`（get/put/batched_\*）/ `DfkvL2Adapter` |
| key 方案 | `{model}/{page_hash}…`（页粒度；MLA 单对象） | chunk 粒度 `@sg` 合并 key（源自 vLLM block hash） | `{model}@{ws}@{wid}@{chunk_hash}`（chunk 单对象） |
| 零拷贝 | 两端零拷贝（GET 直落 HiCache 宿主页，**host-host**） | **GPUDirect RDMA**（KV 直读写 GPU 显存，无 host bounce） | host-host 零拷贝（LMCache pinned arena 一次注册 MR） |
| 块大小 | 固定页（page_size token） | 变长 chunk（SG 多层段合并为一 key） | **任意**（含变长不满末块，走 `GetAuto`） |
| 典型场景 | SGLang PD 生产（GLM-5.1/5.2 MLA） | vLLM 生产直连（DeepSeek-V4-Flash 多池已验证） | vLLM+LMCache 栈；MP-server 路径给多 KV-group 模型 |

三者共用同一套 `libdfkv.so`（ctypes 加载的 C ABI）、value header、传输层与 MDS 发现；
key 命名空间互不相交（详见 §5 共池与隔离）。三条路径都是**纯 Python 对接、无需编译
CPython 扩展**。

**版本兼容（v1.7.x）**：v1.7.0 的服务端新特性 —— slab 存储引擎（`--store-engine`）、
RAM 热层（`DFKV_RAM_TIER`）、io_uring GET（`DFKV_SERVER_URING`）—— **全部是服务端内部实现**，
wire 协议、返回字节、客户端 API 完全不变；客户端**无法分辨**节点跑哪个引擎，也**不需要任何
配置**。v1.6.x 客户端可原样对接 v1.7.x 服务端。

**版本兼容（v1.34 / v1.35）**：同样零客户端改动，但对接方应知道两件事——
① **v1.34 服务端多轨 anchor**（`--rdma-dev` 逗号列表）：客户端按 §1.2 配轨亲和后才吃满
8 轨；推广顺序是**先升服务端**再放开客户端多轨（pre-1.34 服务端对非默认轨无 anchor，
空闲回收后下一波请求付串行重注册风暴）。
② **v1.35 读侧 convoy 合并 + RAM 晋升**（服务端 opt-in `DFKV_READ_COALESCE=1`，见根
README "Recommended tuning"）：TP-N 各 rank 独立进程重复读同页时，服务端把 N 次盘读
合并/晋升——客户端观察到的效果是**同页重复冷读与重放显著变快**（xb01 实测每重复页盘读
8→~2.4 次、晋升页复读零盘），无任何客户端配置或行为变化。

---

## 1. 通用客户端配置（跨连接器 env / config 总表）

以下 env 由 `libdfkv.so`（C 客户端）与各连接器读取，**每个引擎进程都要设**（vLLM/SGLang
的每个 DP rank 是独立进程）。

### 1.1 连接与发现

| env | 默认 | 说明 |
|-----|------|------|
| `DFKV_LIB`（或 `DFKV_BUILD`） | — | `libdfkv.so` 绝对路径（`DFKV_BUILD` 指目录，取 `$DFKV_BUILD/libdfkv.so`）。连接器 config 里的 lib 键优先于 env：HiCache=`lib_path`、vLLM=`lib`、LMCache=`remote_storage_plugin.dfkv.lib` / L2 `lib`。 |
| `DFKV_MEMBERS` | — | **遗留静态**成员表 `name=ip:port,...`，单节点/简单部署用。生产优先 MDS 发现。 |

**MDS 动态发现（生产推荐）**走连接器 config 而非 env：`mds_endpoints=ip:port,...` +
`mds_group=<group>`（须与 `dfkv_server --group` 一致）。客户端后台轮询 MDS
（`mds_poll_ms`，默认 3000ms），etcd epoch 变化时重建加权 Ketama 环，节点增减**无需重启
推理引擎**。各连接器的具体写法见 §2/§3/§4。

**客户端注册**（"谁在用 dfkv"）：三条路径在 MDS 发现成功后，自动把本连接器作为消费方
注册到 `/dfkv/v1/groups/<g>/clients/<id>`（与节点成员表隔离，不入放置环）。死掉的连接器
key 在 TTL（30s）内自动过期，无显式反注册、无脏 key。v1.15.0 起三条路径（SGLang HiCache /
vLLM / LMCache）行为一致，默认开，env `DFKV_CLIENT_REGISTER=0` 或连接器 config
`client_register=0` 关闭。SGLang HiCache 的注册信息串为 `type=hicache,model=<m>,tp_size=..,
tp_rank=..,ver=<lib>`（无 `role`——HiCache 是前缀 L3 缓存，无生产/消费角色之分）。观察：
`dfkvctl clients --mds <ep,...> --group <g>` 或 `dfkv_mds_group_clients` 指标。注意只有
**升级过的**客户端才注册，空列表 = "当前消费方都没注册" 而非 "没人用"。

### 1.2 传输（TCP / RDMA）

| env | 默认 | 推荐 | 说明 |
|-----|------|------|------|
| `DFKV_RDMA` | 未设 = TCP | `1` | 选 native-verbs RDMA 传输；未设则 TCP 回退 |
| `DFKV_RDMA_DEV` | — | 最佳：**按 rank 轨亲和**（每进程注入 `ib7s400p{local_rank}`）；简化：本机所有 400G 口逗号列表 + `DFKV_RDMA_NUMA=1` | RDMA 轨；逗号列表 = 多轨。🔴 **必须显式设置且与 server 服务轨同一 IB fabric**。留空 = 客户端拿本机**枚举第一个**设备——多 fabric 主机（计算轨 + 存储轨共存）上第一个设备常是另一张网（如 200G 存储卡 `ib6s200p0`），与 server 的 400G 轨物理不通 → **跨轨 RC 黑洞**：bootstrap/QP 建连全部成功，数据 op 秒级重传超时死亡、连接池熔断后续秒拒（2026-07-20 xb01 实测签名：服务端 conn+1 但零完成、客户端首 op ~3.7s 失败后批量瞬败）。设备名不同的异构场景也一样：选一个**确认在 server fabric 上**的本机设备显式写上，不要赌枚举顺序 |
| `DFKV_REQUIRE_RDMA` | `0` | 生产 `1` | 无 RDMA 设备时启动失败，禁止静默 TCP fallback |
| `DFKV_RDMA_DEPTH` | `1` | **两侧一致** | 单连接在途请求数。🔴 **契约：客户端 depth 必须 ≤ 服务端 depth（同名 env，各读各的）**——超出的在途请求撞 RNR 重试，表现为**静默 3-4 倍劣化而非报错**（2026-07-04 hd05 实测：两侧=32 时单连接 batch GET 1.39→4.42 GB/s、batch32 1.31→5.81）。⚠️ 历史"depth-flat（1≈8≈16≈32）"结论即此错配的假象（当时服务端恒为默认 1），作废。服务端每连接 pinned 内存 ≈ 2×control_cap×depth，抬 depth 前核内存；多连接 fan-out（`batch_concurrency`）仍是首选吞吐杠杆。服务端现值经 `dfkvctl ring` INFO `qd=` 可查 |
| `DFKV_RDMA_MAX_BLOCK_BYTES` | 0(不声明) | **DCP1 容量声明**（v1.11.0+，issue #110）：本进程在此连接上会 PUT/GET 的最大块字节数。客户端把它藏进 bootstrap 设备名帧的确定性零尾部；新服务端按声明精确分配该连接的 per-slot 缓冲（不再按全局最坏 64MiB），GLM 4MiB 块连接 ≈ **6.7× 内存↓**。连接器可按块几何自动推导后 setenv。超声明的 op 客户端侧直接返回失败（不上 wire）。四配对兼容：不声明/老版本=旧行为。服务端日志 `rdma conn caps: declared=…` 是生效凭证。 |
| `DFKV_RDMA_NUMA` | `0` | 多 NUMA 大机 `1` | 绑 buffer/线程到轨的 NUMA 节点 + 建连时按调用线程 NUMA 选本地轨（无本地轨→轮转全轨）。仅新建连接触发，热路径零开销。SGLang/vLLM 通吃 |
| `DFKV_RDMA_MAX_PAYLOAD_BYTES` | 64 MiB（67108864） | — | 客户端单 value payload 上限（不得超过 server 侧同名上限） |

#### 1.2.1 `DFKV_RDMA_MAX_BLOCK_BYTES` 怎么定（含 L2 / L2-bypass 两套公式）

这个值决定**服务端**每连接 pin 多少内存：`qd × (ValueHeader + 声明值)`，握手时**预先分配**，
不随实际流量增长。B200 实测（qd=32）：声明 16 MiB → **1.02 GB/连接**；声明 4 MiB → **0.34 GB/连接**；
不声明（走 64 MiB 兜底）→ **约 2 GB/连接**。一个 8-rank 推理实例对单台服务端开约 **135 条连接**，
所以这不是微调项，它决定环节点装不装得下。

**块大小取决于走哪条路径**——两条路径的分块规则不同：

| 路径 | 是否 `@sg` 分块 | 公式 | GLM-5.2 实测 |
|---|---|---|---|
| **原版 L2**（host 池） | 否，每对象一块连续内存 | `层数 × page_size × 每token每层字节 / sub` | 78×64×576 = **2,875,392 B (2.74 MiB)** |
| **L2-bypass**（device 直连） | 是，按 SGE 宽度切 | `min(sg段宽, 层数) × page_size × 每token每层字节` | 29×64×576 = **1,069,056 B (1.02 MiB)** |

- `每token每层字节` = MLA 取 `(kv_lora_rank + qk_rope_head_dim) × dtype字节`；GLM-5.2 fp8 = `(512+64)×1 = 576`
- `sub` = MLA 为 1（latent 单对象）、MHA 为 2（k/v 各一）
- `sg段宽` = `min(kMaxSge, HCA max_sge) − 1`（SGE0 让给 header）；当前 `kMaxSge=30` 硬编码 → 29
- 78 层按 29 切成 3 段（29/29/20），最小段 20×36,864 = 737,280 B——两个值均已逐字节实测吻合

**与这些无关**（常见误解）：`BatchGet` 并发（放大的是块的**数量**）、上下文长度（1M 上下文只是页更多，
每页仍切出同样大小的块）、HCA `max_sge`（被 `kMaxSge=30` 压住，换宽网卡不会变大）。

**会改变它的**：`--page-size`（线性）、kv-cache dtype（fp8→bf16 翻倍）、模型 MLA 维度与层数、
以及改动 dfkv 的 `kMaxSge`。

**换模型的 tuning 步骤**：
1. 按上表算出理论值（两条路径都算，取大者——同一集群可能两种都跑）
2. 起一轮真实负载，读服务端/客户端日志里的 `rdma: max block observed <N>B` 高水位（v1.40+）
3. 取实测值的 2~4 倍设定，注意**必须同时覆盖原版 L2 路径的整页对象**
4. 复核服务端日志 `rdma conn caps: declared=…` 确认生效（这行只在客户端真声明时出现）

🔴 **设小了不会报错——这是最危险的部分。** 超声明的块被判 `kInvalid`，而 `kInvalid` 被客户端健康
计数**刻意忽略**，上层看到的就是 `hits[i] != 1`，与"这页压根没缓存"**完全无法区分**：不崩、不报错、
不熔断，只有命中率悄悄封顶。v1.40+ 起超限会打 `rdma: block …B exceeds the declared bound …B` 告警
（首次 + 每 1024 次），在此之前**没有任何信号**。典型踩法：照 L2-bypass 实测的 1.02 MiB 调到 2 MiB，
切回原版 L2 后 2.74 MiB 的整页对象全部静默失效。

> **服务端侧上限 `--max-msg`（v1.40+）**：默认 64 MiB，即"客户端不声明时给多少"。
> 它同时是本服务端接受的**上限**：客户端声明**高于**它会被**明确拒绝连接**并打日志，
> 而不是悄悄按小的开——后者会让客户端按自己声明的大小发包、打爆对端 recv buffer（RNR/QP 断）。
> 大集群建议显式设定，否则服务端的内存预算完全由客户端决定，而客户端常由别的团队部署、版本不一。

> ⚠️ 旧 `rail_affinity`（extra_config）**已废弃为 no-op**（v1.2.0）：它按 `tp_rank`
> 收窄选轨，但 DP-attention 下每 rank `tp_rank=0`→塌缩单轨。配了只打 stderr 告警。
> 用 `rdma_numa` / `DFKV_RDMA_NUMA` 替代。

### 1.3 wire 协议版本

wire 协议为 **v1 单版本**（版本字节在帧首，未知版本 fail-fast 断连）。
v1.7.0/v1.7.1 曾附带实验性 opt-in "wire v2"（`DFKV_WIRE_VERSION=2`，请求 seq 回显校验），
**v1.7.2 起已移除**：生产数据面走 RDMA（RC 硬件已保证按序 + ICRC，v2 冗余且从未在 RDMA
生效），TCP 侧也无部署使用。设置 `DFKV_WIRE_VERSION` 现在无任何效果，可从环境中清理。

### 1.4 块身份（96-bit，无需配置，仅需知道）

v1.7.0 把块 key 加宽到 **96 bit**（`id`=MD5[0..8)，`index`=MD5[8..12)），同模型哈希碰撞
概率降到可忽略。`libdfkv` 内部自动完成——**无 env、无 config**。唯一后果：**v1.6.x 客户端与
v1.7.x 客户端对同一逻辑块算出不同 key**，两版客户端混跑同一环时互不复用对方缓存（干净的
跨版本 miss→重算，绝不脏读）。同版客户端内部完全一致；路由用独立哈希，混版**不会错路由**。

### 1.5 连接器调优 env

| env | 默认 | 说明 |
|-----|------|------|
| `DFKV_CONNECTOR_ID` | — | 逻辑客户端 id，进指标 label |
| `DFKV_CONNECTOR_BATCH_MAX_KEYS` | 连接器各异（LMCache 512） | 单次 native 批量最大 key 数 |
| `DFKV_CONNECTOR_GET_PARALLELISM` | 连接器各异（LMCache 1） | 并发 batched-get 组数（=线程池 worker 数），提高可降 TTFT |
| `DFKV_CONNECTOR_ASSUME_EXISTS` | `0` | 跳过 load 前的 Exist 探测（省一次探测、换可能 miss；调试用） |
| `DFKV_TP_RANK` | — | tensor-parallel rank（MLA 场景仅 rank 0 写） |
| `DFKV_READ_SHARD_KEYS` | `16` | 每读分片的目标 key 数：把单节点的一组批量 GET 切成多分片并发（少节点环/大 batch 集中单节点时突破 ~166 MB/s/conn 的单连接串行 drain 天花板；宽环上无感） |
| `DFKV_READ_MAX_CONNS` | `8` | 单节点读分片的并发连接上限（与上一条配对；`1` = 关闭分片） |
| `DFKV_FANOUT_THREADS` | `32` | 客户端批量操作 fan-out 线程池上限（clamp [1,1024]）；高并发引擎（callers × node-groups ≫ 32）不调会退化 caller-serial，per-call 延迟从 max(group) 变 sum(group) |

以上三条为 **native C 客户端**（`libdfkv.so`）knob，对 HiCache / LMCache / vLLM
三条接入路径同等生效；生产漂移排查时先看启动 config dump（v1.37+ 全 knob 带来源打印）。

### 1.6 可观测性（opt-in，全部不占数据路径）

三条路径共用同一套观测设施；差别只在**配置入口**：HiCache 插件支持 extra_config 键
**或** env（extra_config 优先），vLLM / LMCache 连接器的 telemetry **只认 env**。

| 层 | 打开方式 | 详见 |
|----|---------|------|
| **逐操作访问日志**（一行一 op：`<op>(<args>) : <result> <秒>`，如 `batch_get_auto_sg(20 keys) : hits=20/20, 1310720 bytes <0.007234>`；关 ≈100ns 空操作，开 = 异步落盘、热路径 ~µs） | `DFKV_ACCESS_LOG_ENABLED=1`、`DFKV_ACCESS_LOG_PATH`（空=stderr）、`DFKV_ACCESS_LOG_THRESHOLD_US`（只记 ≥N µs 的 op，0=全记）、`DFKV_ACCESS_LOG_MAX_BYTES`/`_BACKUP_COUNT`（滚动） | [access_log.md](access_log.md) |
| **车队指标 push（OTLP→Collector→Grafana）**：命中率/吞吐/op 延迟 + 逐 peer 延迟 | `DFKV_METRICS_ENABLED=1` + `OTEL_EXPORTER_OTLP_ENDPOINT=http://<collector>:4317`；可选 `DFKV_METRICS_EXPORTER`（`stdlib` 默认零依赖 / `otel`）、`DFKV_METRICS_EXPORT_INTERVAL_MS`（10000）、`DFKV_PROBE_INTERVAL_MS`（空闲也出逐 peer 延迟）、`DFKV_CLIENT_STATS_POLL_S`、`DFKV_PEER_LATENCY_POLL_S` | [METRICS.md](METRICS.md) §3.4、[deploy/observability/CONNECTOR-USAGE.md](../deploy/observability/CONNECTOR-USAGE.md) |
| **分布式追踪 push（OTLP /v1/traces→Jaeger/Tempo）**：慢请求/采样/失败 span | `DFKV_TRACING_ENABLED=1` + `OTEL_EXPORTER_OTLP_ENDPOINT=http://<collector>:4318`；`DFKV_TRACE_SLOW_REQUEST_MS`（1000，0=关）、`DFKV_TRACE_SAMPLE_PERCENT`（0）、`DFKV_TRACE_EXPORT_INTERVAL_MS`（5000）、`DFKV_TRACE_MAX_BUFFERED_SPANS`（2048） | [tracing.md](tracing.md) |
| **总开关** | `DFKV_TELEMETRY_ENABLED=1` = 同时打开 metrics + tracing（各自的显式开关优先） | — |

访问日志三条路径**同一套 env、同一行格式**，一处设置全体生效。多 DP rank 建议
`DFKV_ACCESS_LOG_PATH` 带 rank 后缀区分，或留 stderr 随引擎日志走。

### 1.7 这些**不是**客户端配置（常见混淆）

以下只设在 `dfkv_server` / 其 systemd unit 上，客户端设了**毫无作用**
（见 [ARCHITECTURE.md](ARCHITECTURE.md) §5–7）：

| flag / env | 侧 | 作用 |
|------------|----|------|
| `--store-engine=file\|slab` / `DFKV_STORE_ENGINE` | **server** | 该节点的盘上存储引擎 |
| `DFKV_RAM_TIER` / `DFKV_RAM_TIER_BYTES` | **server** | 写穿 RAM 热层 |
| `DFKV_SERVER_URING` | **server** | io_uring 异步 GET serve 路径 |
| `DFKV_SLAB_WRITE` | **server** | slab I/O 模式（默认 direct；`buffered` 为退出开关） |
| `DFKV_READ_COALESCE` / `_RECUR_MS` / `_TIMEOUT_MS` | **server** | v1.35 读侧 convoy 合并 + RAM 晋升（见根 README "Recommended tuning"） |

混合车队（部分节点 slab、部分 file；部分带 RAM 层）对所有客户端**完全等价**，引擎可逐节点自由选。

---

## 2. SGLang HiCache 对接

把 `libdfkv.so` + `dfkv_hicache.py`（发布包内 `python/`，仓库源在 `integration/hicache/`）
放到 pod 可访问路径，免 fork、`dynamic` 侧载。
前提：GLM-5.1 = MLA（每页 KV ≈ 2.74 MiB 单对象、跨 TP 复制、仅 tp_rank0 写）。

### 2.1 环境注入

启动 `sglang serve` 前注入：

```bash
export PYTHONPATH=/userdata/dfkv:$PYTHONPATH
export DFKV_LIB=/userdata/dfkv/libdfkv.so
export DFKV_RDMA=1                       # 启用 RDMA 数据面（否则 TCP）
export DFKV_REQUIRE_RDMA=1               # 可选：禁止悄悄 TCP fallback
# 数据面设备；多轨用逗号列表（标准节点 8×400G）
export DFKV_RDMA_DEV=ib7s400p0,ib7s400p1,ib7s400p2,ib7s400p3,ib7s400p4,ib7s400p5,ib7s400p6,ib7s400p7
export DFKV_RDMA_NUMA=1                   # 可选：多 NUMA 大机 NUMA 选轨（§1.2）
export DFKV_RDMA_MAX_PAYLOAD_BYTES=67108864  # 可选：单 chunk payload 上限，默认 64MiB
# DFKV_RDMA_DEPTH 两侧一致（§1.2：客户端 > 服务端 = RNR 静默劣化）
```

> ⚠️ hd04 当前只有 `ib7s400p0,ib7s400p1` 两轨 up，但标准训练计算网节点是 8×400G，
> 按本机实际 up 的口列全。

传输相关 env 亦可走 extra_config（`dfkv_open` 前自动设 env，extra_config 优先）：
`"rdma_depth":K`、`"require_rdma":1`、`"rdma_numa":1`。

### 2.2 SGLang 启动 + 后端配置

**方案 A — MDS 动态发现（推荐）**：配 `mds_endpoints` + `mds_group`；插件内部调
`dfkv_start_mds_discovery` 自动轮询 MDS，epoch 变化时重建环，无需重启。

```bash
sglang serve ... \
  --enable-hierarchical-cache --hicache-write-policy write_through \
  --hicache-mem-layout page_first_direct --hicache-io-backend direct \
  --hicache-storage-prefetch-policy timeout \
  --hicache-size <字节> \
  --hicache-storage-backend dynamic \
  --hicache-storage-backend-extra-config '{
    "backend_name":"dfkv","module_path":"dfkv_hicache","class_name":"DfkvHiCache",
    "interface_v1":1,
    "mds_endpoints":"10.0.0.1:9400,10.0.0.2:9400",
    "mds_group":"default",
    "model_hash": 81, "page_size":64, "dtype_tag":1178092852,
    "layer_num":78, "head_num":1, "head_dim":576 }'
```

**方案 B — 静态成员表（遗留）**：无 MDS 时用 `members` 字段，节点增减需重启 SGLang：

```json
"members":"n57=192.168.1.57:28001,n58=192.168.1.58:28001"
```

（其余字段同方案 A，去掉 `mds_endpoints`/`mds_group`。）

**extra_config 全部键**（源 `dfkv_hicache.py`）：`interface_v1`（必填）、
`mds_endpoints`/`mds_group`（默认 `default`）/`mds_poll_ms`（3000）或 `members`、
`model_hash`（0）、`page_size`（64）、`dtype_tag`（0）、`layer_num`/`head_num`/`head_dim`（0）、
`lib_path`、`batch_concurrency`、`rdma_depth`/`require_rdma`/`rdma_numa`、
`client_stats_poll_s`（10s，`0`=关）、访问日志/telemetry 键（`access_log`、`access_log_path`、
`metrics`、`tracing`、`otlp_endpoint`、`trace_slow_request_ms`、`trace_sample_percent` 等，
env 同义项见 §1.6）、`rail_affinity`（已废弃 no-op）。

### 2.3 HiCache 关键 flag

| flag | 推荐 | 说明 |
|---|---|---|
| `--hicache-storage-backend dynamic` | 必填 | 侧载 dfkv 插件，免 fork SGLang |
| `--hicache-io-backend direct` | `direct` | O_DIRECT 零拷贝读写路径 |
| `--hicache-mem-layout page_first_direct` | `page_first_direct` | 配合 direct 后端的内存布局 |
| `--hicache-write-policy write_through` | `write_through` | 写穿，L3 与 L1/L2 同步 |
| `--hicache-storage-prefetch-policy timeout` | `timeout` | prefetch 用 timeout 策略（比 best_effort 更安全的杠杆） |
| `--hicache-size <字节>` | 按 L2-L1 容量 | L3 prefetch 容量；⚠️ 须满足 L2>L1 硬约束，prefetch 容量比建议 0.8 |

### 2.4 必知项与陷阱

- **`interface_v1:1` 必填**，插件 `__init__` 强校验：缺失即 `raise ValueError` 启动失败。
  原因——对 `dynamic` 后端，SGLang 仅在 `interface_v1` 为真时才走零拷贝
  `batch_set_v1/get_v1`；否则退回 generic `set/get`，而 dfkv 的 generic `get/batch_get`
  是未实现的桩 → **写成功、L3 读静默失败**（线上踩过：launch 脚本漏配，14GB 写入但
  prefetch 全 miss）。`interface_v1:1` 下 GET payload 经 RDMA 散射**直落 HiCache 宿主页**
  （client 零拷贝），server O_DIRECT 直读入已注册 direct buffer 并 scatter-send
  （server 无 payload memcpy），两端零拷贝。
- MLA 下插件自动单对象、无 rank 后缀、`backup_skip`（仅 tp_rank0 写）。decode 共享前缀配同 members。
- **多池模型**（Mamba/SWA/DeepSeek-V4）用 v2 PoolTransfer 接口（插件已实现）。
  DSA/DeepSeekV4 主 `kv` 池是无数据的 LogicalHostPool（`get_page_buffer_meta→None`），
  插件对其 `batch_set_v1` 写空 marker 锚定命中前缀、`batch_get_v1` no-op，真实 KV 走 v2 侧池。
- ⚠️ **`page_size` 是几何身份字段，切勿中途改（改了 = 冷缓存）**：它只进 ValueHeader 做
  命中身份校验（`HeaderMatches` 两端必须一致，否则**静默 miss**），**不参与算 value 大小**
  （大小走 `payload_len`=`get_page_buffer_meta` 真实字节）、也**不参与算 key**（key 用
  SGLang 的 page_hash）。含义：① DeepSeek-V4/DSA 等模型 SGLang 会把内部 page_size 强制成
  256，但 extra_config 的 `page_size` 只要**全实例统一**就能正确命中（线上 dsv4-flash 配 64，
  功能正确，仅标签名不实）；② 千万别中途 64↔256 改来改去、或不同实例配得不一样——老数据
  几何不匹配会**静默全 miss（整池冷启）**。真要改：所有共享同 keyspace 的实例**同时切** +
  接受一次冷缓存；`dfkvctl` 手动读时 `--page_size` 也须跟 writer 一致。
- **客户端指标（pull）**：插件自动在 SGLang 自带 `/metrics` 上暴露
  `dfkv_client_*{tp_rank}`（set/get 量、命中、IO 错误、peer 熔断切换、延迟直方图）。
  后台轮询线程读 C 客户端快照，间隔 extra_config `client_stats_poll_s`（默认 10s，
  `DFKV_CLIENT_STATS_POLL_S` 兜底，`0`=关）。全指标见 [METRICS.md](METRICS.md) §3.3。
- push 指标 / 追踪 / 访问日志：见 §1.6（HiCache 走 extra_config `"metrics":1,"otlp_endpoint":...`
  或 env 均可）。

### 2.5 验证

在**一个受控 SGLang 副本**上切 `dynamic` 后端，发共享长前缀请求看命中上涨，确认后推广：
- SGLang `--enable-cache-report` 的 HiCache storage hit/miss、TTFT。
- server 侧 `dfkvctl stat --all --mds <eps> --group <g>` 或 `/metrics` 看 get 命中、写入量
  （见 [DEPLOY.md](DEPLOY.md) §监控）。
- 回滚：`--hicache-storage-backend` 改回原后端（mooncake 等）重启该副本，与 dfkv 解耦。

### 2.6 特性边界（HiCache 侧不适用的 vLLM 特性）

后续给 vLLM 连接器加的几个特性**不适用于 HiCache 侧**——不是漏配，是数据形状不匹配，
HiCache 维持常规路径即可：

- **scatter-gather（SG，合并 key）— HiCache 不用。** SG 把"一个 chunk 的多个层段"合成
  一个多-SGE RDMA key，是为 vLLM 连接器的变长 chunk × 多层段做的。HiCache MLA **每页就是
  一个打包 latent 对象（~2.74 MiB）**，本来一页一 key、无碎段可合，SG 无收益。
  （仅 MHA 的 `_k`/`_v` 对或未来多池 HiCache v2 才理论上有边际收益，且需改插件代码、非开关。）
- **io_uring async GET（`DFKV_SERVER_URING`，server 侧）— 不要开。** 实测单盘对吞吐
  **flat/无收益**（瓶颈是盘不是读提交），默认关。
- **`DFKV_RDMA_DEPTH` — 两侧保持一致**（§1.2：客户端超过服务端 = RNR 静默劣化；历史 depth-flat 结论已作废）。
- HiCache 真正需要的硬化与可观测性（RDMA 空闲回收、CLOCK 持久指针、MDS 硬化、
  Prometheus 指标）**已在 v1.5.2+ 内**，无需额外动作。

---

### 2.7 L2-bypass（L1↔L3 device 直连，绕过 host 池）

把 SGLang 的 L2（host pinned 池）从数据路径上摘掉，GPU 显存与 dfkv 之间直接 GPUDirect RDMA。
省掉一次 device↔host 拷贝和整个 host 池的驻留内存，代价是块被 SGE 宽度切成 `@sg{n}` 子键。

#### 客户端（SGLang 侧）

| env | 值 | 说明 |
|---|---|---|
| `SGLANG_HICACHE_L2_BYPASS` | `1` | 总开关。关闭即回到原版 host L2 路径 |
| `SGLANG_HICACHE_L2_BYPASS_DEDUP` | `1` | 同前缀并发 SG GET 去重：后到的请求 park 等待，不重复拉取 |
| `SGLANG_HICACHE_L2_BYPASS_FUSE_DRAFT` | `0` | draft（EAGLE）是否与目标层融进同一次 RDMA op。**默认关**——收益未经重复取样确认，单次测量不足以认领 |
| `DFKV_RDMA_MAX_BLOCK_BYTES` | 见 §1.2.1 | bypass 下块 = `29 × page_size × 每token每层字节`，比原版 L2 小约 2.7× |
| `DFKV_RDMA_IO_MS` | `30000` | bootstrap 握手超时。**冷连接池场景必须放宽**，理由见下 |

SGLang 启动侧需配合：`--hicache-mem-layout page_first_direct --hicache-io-backend direct`。

#### 🔴 冷连接池：为什么 `DFKV_RDMA_IO_MS` 默认的 10s 不够

服务端每条**新**连接的 `ep.Open()` 要预分配并注册 `qd × (ValueHeader + 声明值)` 的 pin 内存，
这一步占握手耗时的 **98%**（B200 实测六段计时：`devread=147ms open=9046ms qpx=0 connect=0 poolmr=0`）。
新实例的**第一次** L3 读时连接池是空的，要一次性建大量连接，每条都付这个代价：

| 配置 | `open` | 总握手 | 对 10s 超时余量 | 结果 |
|---|---|---|---|---|
| 未声明块大小（64 MiB 兜底） | 9046 ms | 9194 ms | 1.09× | **首读全盘 IO 错误 → L3 读命中 0%** |
| 声明 16 MiB | 5180 ms | 5330 ms | 1.88× | 仍偏紧 |
| 声明 4 MiB | 2837 ms | 2957 ms | 3.4× | — |
| 声明 4 MiB + `IO_MS=30000` | 2837 ms | 2957 ms | **10×** | 稳定 |

失败形态很有欺骗性：握手在 QP 信息交换处超时（`errno=11 EAGAIN`）→ `Acquire` 返回 nullptr →
`MarkBad` → 该轮全部 op 报 IO 错误 → 上层判为全 MISS。**只有首读崩，后续轮次复用池中连接不再走
`Open`，恢复 100% 命中**——这个"只有第一次不行"的形态曾被误判成存储/键方案问题，实际在传输握手。

修复前后（A 构型：新实例 → warmup → seed 写 → flush → 首读）：**8/8 全崩 → 6/6 全 100.0%**，
32 轮长稳 32/32 无 FAIL、无泄漏。

#### 服务端

L2-bypass 不需要服务端改配置，但两项与内存直接相关：

- `--max-msg`（v1.40+）：见 §1.2.1。客户端不声明时的兜底 = 每连接 ~2 GB，大集群务必显式设定
- `--rdma-depth`（qd）：每连接内存与它**线性**相关。B200 实测 32→8：内存 +137GB→+34GB（4.03×），
  握手 `open` 4930→1314ms（3.75×），吞吐中位 −9%（n=4，两臂分布有重叠，只够看方向）。
  内存不紧张时保持 32；紧张时 8 是机制清晰、收益线性的杠杆

#### 验证清单

```bash
# 1. bypass 真的开了（数自证日志，别只看 env）
nerdctl logs <容器> 2>&1 | grep -c 'L2-bypass ENABLED'
# 2. 声明真的到了服务端（这行只在客户端真声明时出现）
journalctl -u dfkv-server | grep 'rdma conn caps: declared='
# 3. 块大小与余量（v1.40+）
nerdctl logs <容器> 2>&1 | grep 'max block observed'
nerdctl logs <容器> 2>&1 | grep -c 'exceeds the declared bound'   # 必须为 0
# 4. 握手耗时（服务端 v1.40+ 诊断构建）
journalctl -u dfkv-server | grep BOOT-SLOW
```

## 3. vLLM 直连 — DfkvStoreConnector

`DfkvStoreConnector` 是 vLLM `KVConnectorBase_V1` 直连连接器：把 KV cache 经
**GPUDirect RDMA** 直接读写到 dfkv 集群，**绕开 LMCache**，占据与
`MooncakeStoreConnector` 相同的 `--kv-transfer-config` 槽位。生产者和消费者读写同一
共享池，实现跨请求、跨实例、跨重启的前缀复用。

连接器纯 Python（ctypes over `libdfkv.so`），直接对 **GPU 设备指针**做 RDMA：分页 KV cache
经 `dfkv_register_memory` 一次注册（nvidia-peermem 下 `ibv_reg_mr` 产出 GPUDirect MR），
传输不经 host bounce。每 chunk 的多层段经 **scatter-gather 批量 API** 合并成一个 dfkv key
（一次多-SGE RDMA / chunk，而非每层段一次），key/磁盘读数 ~20×↓。

### 3.0 角色与前置条件

| 角色 | 要求 |
|---|---|
| **dfkv 存储节点** | NVMe SSD + 400G RDMA 网卡（IB/RoCE）；跑 `dfkv_server`（可选 `dfkv_mds`+etcd 动态成员） |
| **推理节点** | H100/A100 等 GPU + 同一 RDMA fabric；跑 vLLM（≥0.23.0） |
| **GPUDirect** | GPU 节点须加载 `nvidia-peermem`（`lsmod \| grep nvidia_peermem`），否则 `ibv_reg_mr` 拿不到 GPU MR |
| **KV 可再生** | dfkv 是纯 cache：节点丢失 = miss = 重算，无副本、无对象存储兜底 |

dfkv 与 vLLM 可同机（GPU 节点既跑 server 又跑 vLLM，池化本机 NVMe），也可分离。
安装：把 `libdfkv.so` 拷到推理节点；`pip install -e integration/vllm` 装 `dfkv_vllm` 包。

### 3.1 启动 vLLM

**推荐（生产）：MDS 动态发现** —— 配 `mds_endpoints` + `mds_group`。连接器要求
**`mds_endpoints` 或 `members` 二选一**，设了 `mds_endpoints` 即优先走 MDS。

```bash
PYTHONHASHSEED=0 \                       # ★ 必设，见下，否则跨进程/重启不命中
DFKV_RDMA=1 \
DFKV_RDMA_DEV=ib7s400p0,ib7s400p1,ib7s400p2,ib7s400p3,ib7s400p4,ib7s400p5,ib7s400p6,ib7s400p7 \
DFKV_LIB=/opt/dfkv/libdfkv.so \
vllm serve <model> \
  --tensor-parallel-size 2 --data-parallel-size 4 \
  --kv-transfer-config '{
    "kv_connector": "DfkvStoreConnector",
    "kv_connector_module_path": "dfkv_vllm.connector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "mds_endpoints": "192.168.0.8:28150,192.168.0.9:28150,192.168.0.10:28150",
      "mds_group": "glm",
      "model_hash": "1234567890",
      "batch_concurrency": "8"
    }
  }'
```

**备选（单节点/简单部署）：静态成员表** —— 无 MDS 时改用 `members`，节点增减需重启：

```json
"members": "n1=192.168.1.1:28001,n2=192.168.1.2:28001"
```

> ⚠️ **`members` 端口必须是 server 的 `--rdma-port`（RDMA QP bootstrap 监听口），
> 不是主 `--port`。** 指错则每个 RDMA `put` 失败 `rc=-1`。

> **`PYTHONHASHSEED=0` 是头号坑。** dfkv key 的 chunk_hash 源自 vLLM 的 block hash，
> 而 vLLM 0.23.0 的块哈希用 Python `hash()`——默认每进程随机化。DP 各 rank 是**独立进程**，
> 不固定 seed 则同样 token 在不同 rank/重启后算出**不同 key**，跨进程/跨重启复用静默掉到
> ~0（写成功、读永不命中）。每个 rank 都要设，且全实例一致（任意固定值均可，惯例 0）。

### 3.2 验证

1. **首轮（cold）**：发一个长 prompt，记 TTFT。
2. **重启 vLLM**（或换一个 DP 实例）后**发同一 prompt**：连接器工作则 vLLM 跳过 prefill
   （调度日志 `num_computed_tokens` 接近满、`WAITING_FOR_REMOTE_KVS`），TTFT 大幅下降，
   **输出与 cold 逐字一致**。
3. server 侧 `dfkvctl stat --all` 或 `/metrics` 看 get 命中、写入量。

不命中排查顺序：`PYTHONHASHSEED` → MDS 可达（或静态 `members` 端口是否 rdma-port）→
`nvidia-peermem` → `model_hash`/几何是否一致（§5）。**空环 / MDS 不可达**可直接在 vLLM `/metrics`
上看：`vllm:dfkv_client_ring_members==0`（写无处可去）或 `vllm:dfkv_client_mds_reachable==0`
（[METRICS.md](METRICS.md) §3.5），不必翻客户端日志。

### 3.3 环境变量（每个 vLLM 引擎进程）

通用传输/发现/观测 env 见 §1；vLLM 侧要点：

| env | 默认 | 推荐 | 说明 |
|---|---|---|---|
| **`PYTHONHASHSEED`** | 未设 | **`0`（全 rank/实例一致）** | key 确定性，**不设 = 不命中**（§3.1） |
| `DFKV_RDMA` / `DFKV_RDMA_DEV` | — | `1` / 全轨列表 | §1.2 |
| `DFKV_RDMA_DEPTH` | `1` | 保持 1 | depth-flat（§1.2） |
| `DFKV_RDMA_NUMA` | `0` | 多 NUMA 大机 `1` | §1.2 |
| `DFKV_LIB` / `DFKV_BUILD` | — | so 路径 | 被 extra_config `lib` 覆盖 |
| `DFKV_ACCESS_LOG_*` | 关 | 排查时开 | §1.6；vLLM 侧记 `batch_get_auto_sg`/`batch_put_sg`/`batch_exist`/`register_memory` |
| `DFKV_CLIENT_STATS_POLL_S` | `15` | 默认即可 | 后台轮询把环/MDS 健康镜像到 vLLM `/metrics`（`vllm:dfkv_client_ring_members`/`mds_reachable`/`mds_unreachable_polls_total`/`transport_info`）；`0`=关。见 [METRICS.md](METRICS.md) §3.5 |
| `DFKV_METRICS_ENABLED` / `DFKV_TRACING_ENABLED` | 关 | 按需 | §1.6；vLLM 连接器 telemetry **只认 env**（不读 extra_config） |

### 3.4 `kv_connector_extra_config`

| key | 默认 | 推荐 | 说明 |
|---|---|---|---|
| `mds_endpoints` | — | `ip:port,...`（dfkv_mds 层） | **生产首选**；设了即走 MDS 动态发现，省略 `members` |
| `mds_group` | `default` | 如 `glm` | MDS 成员组名，= `dfkv_server --group` |
| `mds_poll_ms` | `3000` | 默认即可 | MDS 轮询间隔（ms） |
| `members` | —（与 mds_endpoints 二选一） | `n=ip:rdma-port,...` | **端口 = server `--rdma-port`** |
| `model_hash` | `0` | 每模型固定 uint64 | key 命名空间；共享须几何一致（§5） |
| `lib` | env 兜底 | so 绝对路径 | |
| `batch_concurrency` | `8` | **大池可调高到 ≈ 节点数** | 跨节点 fan-out，**真正的吞吐杠杆**（depth 是平的） |
| `load_async` | `True` | 保持 True | 异步 load，走 `WAITING_FOR_REMOTE_KVS`、不占关键路径 |
| `enable_cross_layers_blocks` | `False` | 默认 False | 仅当引擎分页布局层内交错时开 |
| `lookup_rpc_port` | ipc 自动 | 一般不设 | rank0 前缀查询 RPC，仅 socket 名冲突时设 |

### 3.5 按场景的推荐配置

- **单实例 / 单 DP**：`PYTHONHASHSEED=0`（跨重启复用仍需）+ `DFKV_RDMA=1` +
  `batch_concurrency=8` 默认，depth 保持 1。
- **多 DP / 多实例共享池**：所有 rank、所有实例 **`PYTHONHASHSEED` 同值** +
  `model_hash` 相同 + 几何一致（§5）。这是跨 DP 复用成立的前提。
- **大集群 / 宽池**：`batch_concurrency` 提到接近 dfkv 节点数，让一批 KV 在更多节点并行。
- **长上下文（50k+）**：load 带宽随上下文线性增长，单盘会成瓶颈；靠**分布式存储环**
  （多 server、多盘）摊带宽，而非调 depth。首请求 JIT 见 §3.6。

### 3.6 实测结果（hd04 H100 + IB，DeepSeek-V4-Flash，参考）

- **功能**：5 个 kv_cache_group（MLA + 多组 SWA）全部正确 offload，跨重启 + 跨 DP 命中
  （present=1058/1058、failed=0），输出与 cold 逐字一致，vLLM 真跳 prefill。
- **首请求 JIT**：每个 DP rank 的**第一个**请求付一次性 ~2s Triton JIT（resumed-prefill +
  SWA-index kernel）；暖后 12k 上下文 WARM≈2s < COLD 2.7s。在意首 token 延迟就在启动后
  给每个 rank 打一个合成命中预热。
- **SG 合并**：每 chunk 一个 key（而非每层段一个），25392→1242 key（~20×），减少 per-key 磁盘读。
- **depth 平**：裸 GET 单连接 depth 1 = depth 32 ≈ 1.24 GB/s，完全一样。
- **传输层**：裸 GET 8 连接 5.2 GB/s、16 连接 6.2 GB/s（详见 [datapath-perf-notes.md](datapath-perf-notes.md)）。

### 3.7 已知问题 / 排查

| 现象 | 原因 / 解 |
|---|---|
| 写成功但**读永不命中** | `PYTHONHASHSEED` 没设或各 rank 不一致（头号坑）；或 `model_hash`/几何不一致 |
| 每个 RDMA `put` 失败 `rc=-1` | `members` 指了 `--port` 而非 `--rdma-port` |
| GPU buffer 上 `dfkv_get_auto` 段错误 | 单 get 在 CPU 上算 CRC，对显存必炸；连接器只走 zero-copy 的 batch 路径（已内置，别手调单 get） |
| `ibv_reg_mr` 失败 / 无 GPUDirect | GPU 节点没加载 `nvidia-peermem` |
| 首 token 偶发慢 ~2s | 每 DP rank 一次性 Triton JIT（非 bug）；预热可消 |
| 异构 HCA（`max_sge<30`）某些 key 不缓存 | SG 段数客户端固定 29；超限的 key 只 fail 自己（降级重算，siblings 正常），非 corruption |

---

## 4. LMCache 对接

把 vLLM 的 KV cache 经 LMCache 卸载到 dfkv 集群。**两条路径，按 LMCache 运行模式选**：

1. **in-process `remote_storage_plugin`**（§4.1–4.4）：经典单进程 LMCache
   （`LMCacheConnectorV1`），`DfkvConnectorAdapter` + `DfkvConnector(RemoteConnector)`。
2. **MP-server L2 adapter**（§4.5）：LMCache 多进程 server（`lmcache server` +
   `LMCacheMPConnector`），**多 KV-group 模型（GLM-5.1/5.2 DSA、DeepSeek-V4-Flash）的
   唯一可用路径**，走 `dfkv_connector.l2_adapter.DfkvL2Adapter`。

### 4.1 前置条件与安装（in-process 路径）

| 角色 | 跑什么 | 要求 |
|---|---|---|
| **推理节点** | vLLM + LMCache + dfkv connector | GPU；vLLM、LMCache（≥0.4.5）；connector 纯 Python（ctypes 调 `libdfkv.so`），无 ABI 匹配问题 |

网络：推理节点须能 TCP 连到每台缓存节点的 dfkv 端口；走 RDMA 则两端须在同一 IB/RoCE 网络。

```bash
source /path/to/your/vllm-venv/bin/activate
pip install <DFKV_SRC>/integration/lmcache       # 纯 Python wheel，不编译
export DFKV_LIB=<LIBDFKV>                        # 指向部署好的 libdfkv.so
```

### 4.2 LMCache 配置 `lmcache.yaml`

**TCP 版**（最简单，先跑通用这个）：

```yaml
chunk_size: 16            # 每 chunk 的 token 数；dfkv 不限块大小，可调大
local_cpu: false
save_chunk_meta: false
remote_storage_plugins: ["dfkv"]
extra_config:
  remote_storage_plugin.dfkv.module_path: dfkv_connector.adapter
  remote_storage_plugin.dfkv.class_name:  DfkvConnectorAdapter
  # 静态成员：name=ip:TCP端口 逗号分隔；末尾 /<group> 任意填
  remote_storage_plugin.dfkv.url:         dfkv://c1=<CACHE1_IP>:18800,c2=<CACHE2_IP>:18800/g1
  remote_storage_plugin.dfkv.membership:  static
  remote_storage_plugin.dfkv.lib:         <LIBDFKV>
```

**RDMA 版**：只改两点 —— URL 用 **RDMA 端口**（server `--rdma-port`），并给 vLLM 进程加
`export DFKV_RDMA=1`。`DFKV_RDMA_DEV` **必须显式设为与缓存节点服务轨同 fabric 的本机
设备**（同型号节点逗号列全 8 轨；异构/多 fabric 主机严禁留空——留空取枚举第一个设备，
撞上另一张网就是跨轨 RC 黑洞，见 §1.2 🔴）。

**生产推荐 MDS 动态发现**（节点增减自动生效）：`membership` 改 `mds`，URL endpoint 改成
dfkv_mds 层 `ip:port` 列表，组名走 URL 末尾 `/<group>`：

```yaml
  remote_storage_plugin.dfkv.membership: mds
  remote_storage_plugin.dfkv.url:        dfkv://192.168.0.8:28150,192.168.0.9:28150,192.168.0.10:28150/glm
```

### 4.3 启动 vLLM（in-process 路径）

```bash
export LMCACHE_USE_EXPERIMENTAL=True
export LMCACHE_CONFIG_FILE=/path/to/lmcache.yaml
export DFKV_LIB=<LIBDFKV>
export DFKV_RDMA=1        # 只有走 RDMA 才加（TCP 不要加）

vllm serve <model_path> \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
    --port 18200 --host 0.0.0.0
```

**验证**：
1. **连接器已加载**（vLLM 启动日志，且没回退 blackhole）：
   `Discovered adapter: DfkvConnectorAdapter` / `DfkvConnector ready: membership=... endpoint=... rdma_pools=1 ...`
2. **缓存命中**：同一请求发两遍（或同 `--seed` 的 bench 跑两遍），第二遍
   `External prefix cache hit rate` > 0、TTFT 明显下降；也可开 `DFKV_ACCESS_LOG_ENABLED=1`
   看 `batch_set`/`batch_get` 命中。
3. **数据落盘**：缓存节点 `du -sh <数据盘>/dfkv` 增长。

bench 示例：

```bash
vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
  --dataset-name random --random-input-len 16000 --random-output-len 100 \
  --model <model_path> --base-url http://127.0.0.1:18200 \
  --num-prompts 20 --max-concurrency 10 --seed 1109
```

### 4.4 配置项参考（in-process）

`extra_config` 插件键（`remote_storage_plugin.dfkv.*`）：

| 键 | 必填 | 说明 |
|---|---|---|
| `module_path` / `class_name` | 是 | 固定 `dfkv_connector.adapter` / `DfkvConnectorAdapter` |
| `url` | 是 | `dfkv://<endpoint>/<group>`。static 模式 endpoint=`name=ip:port,...`；mds 模式 endpoint=MDS `ip:port` 列表 |
| `membership` | 否 | **`mds`（默认）** 或 `static` |
| `lib` | 否 | `libdfkv.so` 路径（覆盖 `DFKV_LIB`） |
| `mds_poll_ms` | 否 | mds 模式轮询间隔，默认 3000 |

也支持简写 URL 直连（`plugin://dfkv` 场景 URL 即成员串），此时 knob 全走默认
（membership=mds、lib 走 env）。

环境变量（连接器专属；通用项见 §1）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `DFKV_CONNECTOR_GET_PARALLELISM` | 1 | 并行 batched-get 组数（提高可降 TTFT） |
| `DFKV_CONNECTOR_BATCH_MAX_KEYS` | 512 | 单次 native 批量最大 key 数 |
| `DFKV_CONNECTOR_ASSUME_EXISTS` | 0 | 跳过远程 contains 检查（调试） |

### 4.5 MP-server L2 adapter（`LMCacheMPConnector`，多 KV-group 模型）

LMCache 多进程 server 经 `L2AdapterInterface`（而非上面的 in-process
`remote_storage_plugins`）驱动远端层。`DfkvL2Adapter` 实现该接口，经 LMCache 内置
`plugin` L2 adapter 加载：

```bash
# 1) 启动 MP server，dfkv 作远端（L2）层：
lmcache server --port 6555 --max-workers 8 --l1-size-gb 80 \
  --eviction-policy LRU --chunk-size 256 \
  --l2-adapter '{"type":"plugin",
    "module_path":"dfkv_connector.l2_adapter",
    "class_name":"DfkvL2Adapter",
    "config_class_name":"DfkvL2AdapterConfig",
    "adapter_params":{
      "url":"dfkv://<mds_ip:port,...>/<group>",
      "membership":"mds",
      "lib":"/path/to/libdfkv.so",
      "model_name":"<deployment-name>"}}'

# 2) vLLM 指向 MP server（注意 --no-enable-prefix-caching 把全部 KV 复用交给 LMCache）：
vllm serve <model> --tensor-parallel-size 8 --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both",
    "kv_connector_extra_config":{"lmcache.mp.port":6555}}'
```

`adapter_params` 键：`url`（必填，语法同 in-process）、`membership`（`mds` 默认 |`static`）、
`lib`（否则 `DFKV_LIB`）、`model_name`（隔离命名空间 → 稳定 dfkv `model_hash`）、
`mds_poll_ms`（3000）、`page_size`（0 = 关几何守卫）、`num_workers`（8）、
`max_capacity_gb`（0 = 容量交给 dfkv 自管；>0 开 LMCache 聚合 L2 淘汰，见 §4.6.6）。
server 的 pinned L1 arena 在 LMCache 传入 `l1_memory_desc` 时自动注册 RDMA 零拷贝。

实现要点：dfkv 无原生 eventfd，`DfkvL2Adapter` 用**后台 asyncio loop + 三个
`create_event_notifier`（store/lookup/load）+ done-callback** 把同步 ctypes 客户端桥接到
`L2AdapterInterface`；`ObjectKey`→`model_name@kv_rank@group@hash[@salt]`。
已在 GLM-5.2（vLLM 0.23.0 + LMCache 0.4.7）真机验证：store → 重启（L1 清空）→ 从 dfkv
回载、prefill 跳过。单测 `integration/lmcache/tests/test_l2_adapter.py`（fake client）+
集成测试 `test_l2_adapter_integration.py`（`DFKV_L2_URL`/`DFKV_L2_MEMBERSHIP` 指向真环）。

### 4.6 设计与实现

connector 移植自 dingofs 项目的 LMCache connector，与 HiCache 插件走同一套底层 KV 客户端。

#### 4.6.1 整体架构（in-process 路径）

```
 vLLM (LMCacheConnectorV1, kv_both)
   └─ LMCache Engine
        └─ RemoteBackend ──创建──> DfkvConnectorAdapter (plugin://dfkv)
                                      └─ DfkvConnector(RemoteConnector)
                                           ├─ ExistsLRU            (本地存在性短路)
                                           └─ DfkvNativeClient     (ctypes + 线程池)
                                                └─ libdfkv.so (C ABI)
                                                     └─ KVClient ──一致性哈希──> dfkv cache nodes
```

- LMCache 把每个 prompt 切成 `chunk_size` token 的 **chunk**，对每个 chunk 做内容哈希得
  `CacheEngineKey`，再调 `RemoteConnector` 的 get/put/exists。
- `DfkvConnector` 把 `CacheEngineKey` 序列化成 dfkv key 字符串，把 chunk 字节
  （`MemoryObj.byte_array`，LMCache 固定 host arena 的切片）经 `DfkvNativeClient`
  零拷贝交给 dfkv。
- dfkv `KVClient` 按 key 一致性哈希路由到 cache node，走 RDMA（或 TCP）读写。

#### 4.6.2 与 HiCache 插件的异同

| 维度 | HiCache（SGLang） | LMCache |
|---|---|---|
| 接口 | `HiCacheStorage`（batch_set_v1/get_v1…） | `RemoteConnector`（get/put/batched_\*…）/ L2 adapter |
| 底层 | `libdfkv.so`（ctypes）| 同 —— **相同** |
| key | `{model}/{page_hash}_{tp}_{rank}_k/_v`（页粒度） | `{model}@{ws}@{wid}@{chunk_hash}`（chunk 单对象） |
| 块大小 | 固定页 | **任意**（`full_chunk_size_bytes`，可变长） |
| 零拷贝 buffer | SGLang host KV pool | LMCache `MixedMemoryAllocator` arena |

复用同一套 value header、传输层，key 命名空间不同、互不影响。

#### 4.6.3 相对 dingofs 版的两处实质改动

**① 任意块大小（含变长不满块）。** dingofs connector 把块硬编码 4 MiB（固定 io_uring
buffer）；dfkv 无此限制，直接用 LMCache 的 `full_chunk_size_bytes`（可几十 MiB）。
**难点：dfkv 的 GET 是精确大小匹配**——value 存为 `[48B ValueHeader | payload]`，
`KVClient::Get` 仅当 `payload_len ==` 请求的 n 且 geometry 头匹配才算命中。而 LMCache 会存
**变长不满（unfull）末块**，用满块大小去 GET 会 `payload_len != n` 判 miss、永远读不回。
**解法：变长 get**（dfkv C ABI 为此新增）：
- C++：`KVClient::GetAuto(key, out, cap, *out_len)` / `BatchGetAuto(items, *out_lens)`
  （`src/client/kv_client.{h,cc}`）——从存储头读出真实 `payload_len`，`<= cap` 即命中。
- C ABI：`dfkv_get_auto` / `dfkv_batch_get_auto`（`src/client/dfkv_c_api.{h,cc}`）。
- `BatchGetAuto` 复用 `BatchGet` 的 **RDMA 零拷贝 `RangeInto` 路径**：buffer 容量当请求
  长度，server 返回真实 `[header|payload]`，payload 零拷贝散射进调用方 buffer 并回传
  header。**满块仍是零拷贝热路径**（与 HiCache 一致），不满块按真实长度读回；connector
  再调 `reshape_partial_chunk(memory_obj, bytes_read)` 裁剪 shape。

**② pybind11 → ctypes。** dingofs 用 pybind11 原生模块（eventfd 完成队列）；dfkv 直接
ctypes 调 `libdfkv.so`（与 HiCache 插件一致）。C ABI 同步且内部线程安全：`dfkv_batch_*`
阻塞、内部线程池跨 owning node 并行 fan-out；成员 ring 有互斥锁，**一个 `dfkv_open`
handle 可多线程共享**。`ctypes.CDLL` 调用期间释放 GIL，把阻塞调用派发到
`ThreadPoolExecutor` 即得真并发——无需原生 demux 线程或跨线程 Future 桥接。

#### 4.6.4 包结构与各文件职责

```
integration/lmcache/
├── pyproject.toml            # name="dfkv-connector"，纯 Python（py3-none-any wheel）
├── Makefile                  # make lib / wheel / install / test
└── src/dfkv_connector/
    ├── __init__.py           # 导出 DfkvConnector, DfkvConnectorAdapter
    ├── adapter.py            # ConnectorAdapter：dfkv:// URL + extra_config → connector
    ├── config.py             # parse_dfkv_url -> DfkvEndpoint（mds | static）
    ├── remote_connector.py   # DfkvConnector(RemoteConnector)
    ├── native_client.py      # DfkvNativeClient：ctypes + ThreadPoolExecutor
    ├── l2_adapter.py         # DfkvL2Adapter：MP-server L2 路径（asyncio + eventfd 桥）
    ├── key_mapper.py         # CacheEngineKey -> dfkv key 字符串（完整 hash、明文 ws/wid）
    ├── access_log.py         # 逐操作访问日志（默认关，DFKV_ACCESS_LOG_*）
    └── exists_cache.py       # ExistsLRU：「刚 put 又问存在」的远程往返短路
```

`native_client.py` 要点：
1. `load_lib(path)` 声明全部 C ABI 的 `restype/argtypes`（含 `dfkv_get_auto`/
   `dfkv_batch_get_auto`；`dfkv_remove`/`dfkv_batch_remove` 为增量符号，旧库经
   `supports_remove()` 探测）。库路径优先级：显式 `lib` → `DFKV_LIB` → `$DFKV_BUILD/libdfkv.so`。
2. 一个实例一个 handle：`dfkv_open(geometry…)`；mds 模式再 `dfkv_start_mds_discovery`；
   对 `_collect_rdma_pools` 给出的每块区域 `dfkv_register_memory`（一次注册 host arena，
   RDMA 读写其切片即免逐操作 MR 注册）。
3. 专用 `ThreadPoolExecutor(max_workers=get_parallelism)`，`loop.run_in_executor` 派发
   阻塞 ctypes 调用。
4. 零拷贝指针：`(c_char*nbytes).from_buffer(mv)` 直接别名可写连续 buffer；只读 buffer
   退回 `from_buffer_copy`（罕见，正确性优先）。keepalive 对象保活到 C 调用返回（RDMA 写安全）。
5. 返回结构：`batch_set→(ok, per_key)`；`batch_get→(ok, per_key, lengths)`（lengths 给
   reshape 用）；`batch_exists→per_key`。`close()` 先 `executor.shutdown(wait=False)` 再 `dfkv_close`。

`remote_connector.py` 要点：put 按 `len(byte_array)` 真实大小存（不满块也按真实大小写）；
get 分配满块 buffer、变长 get 拿回 `(per_key, lengths)`，命中后满块直接返回、不满块
`reshape_partial_chunk` 裁剪，非法长度当安全 miss 丢弃；`batched_get*` 保持 LMCache
「连续前缀」语义；缓冲分配走 `local_cpu_backend.allocate(...)`（host arena 切片）。

#### 4.6.5 geometry 映射与 key 方案

dfkv 每个 value 头带 geometry（model_hash/page_size/dtype_tag/tp_\*/layer/head…），GET 时
`HeaderMatches` 校验读写两端一致。一个 connector 实例 = 一次 `dfkv_open`、固定 geometry，
故字段是**自一致性标记**，从 LMCache `metadata` **确定性**推导（绝不用随机值，否则重启后
旧缓存读不回）：

| dfkv_open 参数 | 取值 |
|---|---|
| `model_hash` | `blake2b(model_name)` 低 64 位（稳定） |
| `page_size` | `metadata.chunk_size` |
| `dtype_tag` | 0 |
| `flags` | `use_mla ? 1 : 0` |
| `tp_size` / `tp_rank` | `world_size` / `worker_id` |
| `layer_num`/`head_num`/`head_dim` | `kv_shape[0]/[3]/[4]` |

改 model / chunk_size 会让旧缓存 key 失效（chunk_hash 本来也变），预期行为。

key：dfkv key 是不透明字符串、server 按整串哈希分片，渲染为
`{model_name}@{world_size}@{worker_id}@{chunk_hash_hex}`（完整 chunk_hash 不截断、
ws/wid 明文；dtype 不编码——chunk_hash 已是内容哈希，geometry 头兜底）。

成员发现：URL `dfkv://<endpoint>/<group>` 由 `membership` 决定解释——**mds（默认）**
endpoint=MDS 列表，`dfkv_open` 传空 members + 后台 `dfkv_start_mds_discovery`；
**static** endpoint=字面成员串，`<group>` 不用。mds 模式 open 后到首次轮询期间 ring
可能为空，早期操作安全 miss（LMCache 当 cache miss 重算）。

#### 4.6.6 remove / L2 淘汰与边界

- **remove 已支持**（dfkv 已有 `remove` RPC）：L2 adapter 设 `max_capacity_gb > 0` 即启用
  LMCache 的 L2EvictionController，超容量时 `DfkvL2Adapter.delete()` →
  `dfkv_batch_remove` 删块；in-process 的 `remove_sync` 同样由 `dfkv_remove` 支撑。
  需要带 remove RPC 的 `libdfkv.so`/`dfkv_server`（旧库经 `supports_remove()` 探测，
  delete 路径降级为记日志的 no-op）。默认 `max_capacity_gb = 0` = 容量交给 dfkv 各节点
  自己的 LRU。
- **无枚举**：dfkv 无 listing RPC，`list()` 返回 `[]`；`batched_contains` 未实现。
- **RDMA MR 注册**：`_collect_rdma_pools` 取 LMCache `MixedMemoryAllocator.buffer`
  （一块连续 pinned tensor）的 `(addr, length)`，一次注册覆盖全部流量。分页 / P2P
  allocator（`enable_p2p=true`，无单一 buffer）暂不支持——打 warning 并退回逐操作 MR 注册。
- **大 chunk**：dfkv 无 4 MiB cap，但部署前应用真实 `full_chunk_size_bytes` 冒烟一次，
  确认 wire frame 接受大 value（默认上限 §1.2 `DFKV_RDMA_MAX_PAYLOAD_BYTES`=64MiB）。

#### 4.6.7 测试

- **C++ gtest**：`test/client/get_auto_test.cc` —— `GetAuto`/`BatchGetAuto` 覆盖满块、
  不满块、cap 过小（miss）、geometry 不匹配（miss）、双节点批量混合大小。
- **Python 冒烟**：`test/python/dfkv_lmcache_native_smoke.py` —— `DfkvNativeClient` 对接
  本地 `dfkv_server`，put→exists→变长 get（满块+不满块）逐字节校验，无需 torch/lmcache。
- **L2 adapter**：`integration/lmcache/tests/test_l2_adapter.py`（单测，fake client）、
  `test_l2_adapter_integration.py`（真环集成）。

### 4.7 实测结果（参考）

环境：a100（vLLM 0.21 + LMCache 0.4.5）；dfkv = 2 节点 static 成员（TCP 18800 / RDMA
18801）；DeepSeek-R1-Distill-Qwen-32B（TP=1）；`chunk_size=16`（每 chunk ≈4 MiB）；
bench random 16000-in / 100-out，20 prompts，并发 10，同 `--seed`。冷遍写入约 **81 GB**。

| 遍次 | 传输 | External 命中率 | 时长(s) | Mean TTFT(ms) | Median TTFT(ms) | TPOT(ms) | 吞吐(tok/s) | 失败 |
|---|---|---|---|---|---|---|---|---|
| 冷（写入）| TCP | 0% | 181.6 | 62494 | 82321 | 114.1 | 1774 | 0/20 |
| 热（满命中）| TCP | 100% | 126.5 | 42495 | 55981 | 86.5 | 2547 | 0/20 |
| 热（满命中）| **RDMA** | **100%** | **68.7** | **21914** | **28638** | **57.4** | **4688** | 0/20 |

- 缓存复用（RDMA 满命中 vs 冷遍）：Mean TTFT **−65%**，吞吐 **+164%**。
- RDMA vs TCP（同 100% 命中）：Mean TTFT **−48%**，吞吐 **+84%**（TCP 下 KV 加载带宽是
  瓶颈，换 RDMA 后命中收益才完全释放）。
- `chunk_size=16` 每 chunk ≈4 MiB（旧 dingofs 上限），dfkv 正常存取——**任意块大小已验证**。

**纯传输对比**（并发 1、output 1、100% 命中，单请求只测「从 dfkv 加载 16000-token
prompt ≈4 GB KV」的 TTFT）：

| 传输 | Mean TTFT(ms) | Median TTFT(ms) | P99 TTFT(ms) | 时长(s) | 输入吞吐(tok/s) |
|---|---|---|---|---|---|
| TCP | 3842 | 4033 | 4059 | 76.9 | 4166 |
| **RDMA** | **1211** | **1259** | **1311** | **24.2** | **13212** |

即单请求 KV 加载 RDMA ≈1.2s vs TCP ≈3.8s，**TTFT −68%（约 3.2×）**，均 20/20 成功。

### 4.8 已知问题 / 排查

- **vLLM 在请求被中止时崩溃**：LMCache 0.4.5 + vLLM 0.21 在 `FINISHED_ABORTED` 时
  scheduler 进程 `vllm_v1_adapter.request_finished` 会 `assert self.lmcache_engine is not
  None` 崩溃。**与 dfkv 无关**（任何 remote backend 都触发），正常完成请求不走该路径；上游已知。
- **启动日志没有 `DfkvConnector ready`**：查 `lmcache.yaml` 的 `module_path/class_name/url`，
  及 connector 是否装进了 vLLM 用的那个 Python 环境。
- **全部 miss / 连不上**：确认推理节点能连缓存节点端口（TCP 用 `--port`，RDMA 用
  `--rdma-port`）；RDMA 还要确认 `DFKV_RDMA=1` 已设、`DFKV_RDMA_DEV` 指向与 server
  服务轨**同一 IB fabric** 的本机设备（§1.2 🔴）。**建连成功但 op 全部超时/瞬败** =
  跨轨黑洞的典型签名——client 与 server 各自的设备在不同 fabric 上，换对设备即愈。
- **端口被占用**：换一组端口（`--port/--rdma-port` 与 yaml URL 同步改）。

---

## 5. 共池、keyspace 隔离与几何守卫（跨连接器）

**能否共用同一套 dfkv 集群 / 同一个哈希环？可以。** 环只负责把 key 路由到节点；不同引擎、
不同模型指同一 members/MDS 组即可共用存储池与容量（共享各节点 LRU）。

**但不同引擎不会复用彼此的 KV（同模型也不行）。** 三条路径的 key 方案（前缀/哈希/`@sg`
后缀/tp_rank 处理）与 KV 字节布局（MLA 单页打包对象 vs 连接器逐组分段）完全不同——同一
模型、同一段 token，各路径算出的 **key 不同、字节也不兼容**。共存于同一池、keyspace
互不相交，谈不上跨引擎前缀命中。

**几何守卫（同一路径内共享池时务必确认）**：实例 A 写的 KV，只有当张量几何与实例 B 一致时
才能安全被 B 读回。共享 `model_hash` 前确认全部相同：**`--kv-cache-dtype`、page/block
size、KV 内存布局、`--max-model-len`**。dfkv 值头只守 `payload_len`（字节大小）——
**同大小不同布局会被静默读成脏数据**。要隔离就用不同的 `served-model-name` /
`model_hash`（HiCache 与 vLLM 侧取不同值即可保证即使 key 理论撞名也互不串味）。
共池铁律：**共享节点与容量、隔离 keyspace**。

---

## 相关文档

- [DEPLOY.md](DEPLOY.md) — dfkv 集群（server + MDS）标准部署
- [ARCHITECTURE.md](ARCHITECTURE.md) — 存储/协议架构（wire 协议、slab、RAM 热层）
- [datapath-perf-notes.md](datapath-perf-notes.md) — RDMA 数据面性能笔记（depth-flat、SG coalescing）
- [access_log.md](access_log.md) / [METRICS.md](METRICS.md) / [tracing.md](tracing.md) — 可观测性
- `integration/vllm/README.md` · `integration/lmcache/README.md` — 包内快速参考
- `docs/superpowers/specs/2026-06-18-dfkv-vllm-store-connector-design.md` — vLLM 连接器设计文档
