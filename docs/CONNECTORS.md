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
| key 方案 | 页 hash + pool/component/并行坐标 | chunk hash + pool/component/并行坐标，可追加 binary SG 坐标 | chunk hash + pool/component/并行坐标 |
| 零拷贝 | 两端零拷贝（GET 直落 HiCache 宿主页，**host-host**） | **GPUDirect RDMA**（KV 直读写 GPU 显存，无 host bounce） | host-host 零拷贝（LMCache pinned arena 一次注册 MR） |
| 块大小 | 固定页（page_size token） | 变长 chunk（SG 多层段合并为一 key） | **任意**（含变长不满末块，走 `GetAuto`） |
| 典型场景 | SGLang PD 生产（GLM-5.1/5.2 MLA） | vLLM 生产直连（DeepSeek-V4-Flash 多池已验证） | vLLM+LMCache 栈；MP-server 路径给多 KV-group 模型 |

三者共用同一套 `libdfkv.so` C ABI、传输层与 MDS 发现，value 都是无 dfkv
信封的原始字节。自动 namespace 还绑定连接器的 raw-layout ID，默认互相隔离；
对象 key 统一使用 §1.4 的 canonical schema。三条路径都是**纯 Python 对接、
无需编译 CPython 扩展**。

**原生身份/裸值切换是 clean break。** 新 client、server 与 connector 应一起
升级并接受一次冷缓存；不会读取旧 key，也不会双写旧身份或旧 value 格式。

**版本兼容（v1.34 / v1.35）**：同样零客户端改动，但对接方应知道两件事——
① **v1.34 服务端多轨 anchor**（`--rdma-dev` 逗号列表）：客户端按 §1.2 配轨亲和后才吃满
8 轨；推广顺序是**先升服务端**再放开客户端多轨（pre-1.34 服务端对非默认轨无 anchor，
空闲回收后下一波请求付串行重注册风暴）。
② **v1.35 读侧 convoy 合并 + RAM 晋升**（服务端 opt-in `DFKV_READ_COALESCE=1`，见根
README "Recommended tuning"）：TP-N 各 rank 独立进程重复读同页时，服务端把 N 次盘读
合并/晋升——客户端观察到的效果是**同页重复冷读与重放显著变快**（xb01 实测每重复页盘读
8→~2.4 次、晋升页复读零盘），无任何客户端配置或行为变化。

**RDMA transport v2** 是唯一受支持的 native RDMA 协议。client 先做能力
probe，server 要求 DCP2、QP v2 和非零 block 声明；任一协商或共享 receive
segment 建立/注册失败都会明确拒绝启动或连接，不会切换为其他 RDMA 数据面。
TCP 仍使用当前 versioned native wire。

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
（`mds_poll_ms`，默认 3000ms），**成员放置内容 epoch** 变化时重建加权 Ketama
环；该 epoch 是成员内容 hash，不是 etcd 全局 revision，因此无关写入和 stats
heartbeat 不会重建环。节点增减**无需重启推理引擎**。

节点停止后，权威移除须先等 30s lease TTL，再等下一次 poll；若一次缩容超过
`DFKV_MDS_SHRINK_GUARD_PCT`（默认 50%），客户端还要求该视图连续 3 次 poll
才采纳（默认约 TTL+9s）。这段 hysteresis 期间继续使用旧环，传输失败由 peer
cooldown 快速转为 miss。

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
| `DFKV_RDMA` | 一般路径未设 = TCP；**vLLM 直连无默认，必须 `1`** | 按连接器选择 | `1` 显式选择 native-verbs RDMA v2；请求 RDMA 后设备或协议不可用会失败，不会自动选择 TCP。`DfkvStoreConnector` 只接收 GPU 设备指针，构造时会关闭并拒绝任何非 RDMA handle。 |
| `DFKV_RDMA_DEV` | 首个 `ACTIVE` 本地 HCA | 留空让两端各自选本地首口；多轨才显式写同 fabric 白名单 | 留空时 bootstrap 不发送设备名，client/server 可使用不同本地命名。逗号列表显式开启多轨，新连接在健康轨间轮转；显式设备名会发给 peer，故两端必须存在同名且互通的 fabric。 |
| `DFKV_RDMA_DEPTH` | `1` | 两侧可不同，按容量选 | 握手协商 `min(client,server)` 作为安全窗口。每连接注册 `2 × depth × (18 B + 32 KiB)` 的有界 SEND/RECV control buffer，并从共享 receive segment 租 `depth` 个 slot。 |
| `DFKV_RDMA_MAX_BLOCK_BYTES` | 64 MiB 安全上限 | 按连接器块几何精确设置 | DCP2 声明本连接最大 PUT/GET block，决定共享 segment 的 slot 大小；超声明请求在客户端失败且不上 wire。声明越准，同一 segment 可容纳的 live/pooled v2 连接越多。 |
| `DFKV_RDMA_RECV_SEGMENT_SIZE` | 2 GiB | 按下文 live/pooled 连接公式设置 | server 启动时申请，并在每个选中 rail 的共享 PD 上注册；失败会拒绝启动，segment 无可用 lease 时拒绝新连接。 |
| `DFKV_RDMA_NUMA` | `0` | 显式多轨的大机可设 `1` | 建连时按调用线程 NUMA 选本地 rail（无本地 rail→轮转白名单），server serve 线程跟随 QP rail。单块共享 receive segment 不做 per-rail NUMA 分配；仅保证选轨/线程亲和。 |
| `DFKV_RDMA_MAX_PAYLOAD_BYTES` | 64 MiB（67108864） | — | 客户端单 value payload 上限（不得超过 server 侧同名上限） |

**v2 数据面**：PUT 把 `[request prefix | raw payload]` 以
`RDMA_WRITE_WITH_IMM` 直接写入 server 租出的 slot；GET 先用 SEND 提交
`{addr,rkey,len}` 目标描述符，server 再以 RDMA WRITE 直接散射到调用方
buffer，最后只 SEND 状态与 authoritative stored length。两向 block payload
都不经过 control buffer。`kMembers` 在隔离的 control lane 上使用显式
`18-byte prefix + 32-KiB data` 容量；边界值完整返回，更大响应失败而不截断。

#### 1.2.1 `DFKV_RDMA_MAX_BLOCK_BYTES` 怎么定（含 L2 / L2-bypass 两套公式）

这个值在 **v2** 决定共享 receive segment 的 slot 大小：
`align4K(4 KiB + 声明的最大 raw payload)`。每条数据连接租 `depth` 个 slot，
但所有连接共享一块启动期注册的 `DFKV_RDMA_RECV_SEGMENT_SIZE`，不再各自
注册 `depth × block` 的收发 buffer，因此声明保持精确仍有价值。

**共享 segment 容量必须按连接寿命算，不是按同时在飞请求算。** 数据 QP 的
slot 为

```
S_data = align4K(4096 + DFKV_RDMA_MAX_BLOCK_BYTES)
S_control = align4K(4096 + (18 + 32768)) = 40960
B_required >= N_data × depth × S_data + N_control × depth × S_control
```

`N_data` / `N_control` 是该 server 上所有 rank、进程的**峰值 live + client
pool 中空闲连接**；lease 一直保留到 QP 被销毁或 `DFKV_RDMA_IDLE_MS` 回收，
线程峰值留下的 pooled QP 也要计入。4 MiB 声明、depth=4 时
`S_data=4,198,400 B`，2 GiB segment 最多约 127 条 data QP（未扣 control
lease）；depth=8 时约 63 条。上线同时观察
`dfkv_rdma_recv_segment_free_bytes` 与 `dfkv_rdma_v2_ready`；free 接近 0
即扩容或缩小声明/depth/pool，避免新连接被拒绝。

**块大小取决于走哪条路径**——两条路径的分块规则不同：

| 路径 | 是否追加 binary SG 坐标 | 公式 | GLM-5.2 实测 |
|---|---|---|---|
| **原版 L2**（host 池） | 否，每对象一块连续内存 | `层数 × page_size × 每token每层字节 / sub` | 78×64×576 = **2,875,392 B (2.74 MiB)** |
| **L2-bypass**（device 直连） | 是，按 SGE 宽度切 | `min(sg段宽, 层数) × page_size × 每token每层字节` | 29×64×576 = **1,069,056 B (1.02 MiB)** |

- `每token每层字节` = MLA 取 `(kv_lora_rank + qk_rope_head_dim) × dtype字节`；GLM-5.2 fp8 = `(512+64)×1 = 576`
- `sub` = MLA 为 1（latent 单对象）、MHA 为 2（k/v 各一）
- `sg段宽` 以打开客户端后 `dfkv_max_sg_segs()` 返回的 active-transport
  runtime capability 为准；RDMA 当前为 `min(kMaxSge, HCA max_sge) − 1`
  （SGE0 让给 wire request prefix），TCP 则返回该 transport 的能力。禁止在
  connector/API 调用方硬编码 29。下表的 29 是本次实测客户端返回值。
- 该实测宽度下 78 层切成 3 段（29/29/20），最小段 20×36,864 =
  737,280 B——两个值均已逐字节实测吻合。

**与这些无关**（常见误解）：`BatchGet` 并发（放大的是块的**数量**）、上下文长度（1M 上下文只是页更多，
每页仍切出同样大小的块）。**会改变它的**：`--page-size`（线性）、
kv-cache dtype（fp8→bf16 翻倍）、模型 MLA 维度与层数，以及 runtime SG
capability；HCA `max_sge` 低于 dfkv 上限时会缩小宽度，高于上限则不会放大。

**换模型的 tuning 步骤**：
1. 按上表算出理论值（两条路径都算，取大者——同一集群可能两种都跑）
2. 起一轮真实负载，读服务端/客户端日志里的 `rdma: max block observed <N>B` 高水位（v1.40+）
3. 取实测值的 2~4 倍设定，注意**必须同时覆盖原版 L2 路径的整页对象**
4. 复核服务端日志 `rdma conn: protocol=v2 declared=… control=… shared-slot=… qd=…` 确认生效

🔴 **设小后上层仍只看到 miss——这是最危险的部分。** 超声明的块被判 `kInvalid`，
而 `kInvalid` 被客户端健康计数刻意忽略，上层 `hits[i] != 1` 与“这页压根没缓存”
无法区分：不崩、不熔断。v1.40+ 会打 `rdma: block …B exceeds the declared bound
…B` 告警（首次 + 每 1024 次），所以必须纳入日志告警。
典型踩法：照 L2-bypass 实测的 1.02 MiB 调到 2 MiB，切回原版 L2 后
2.74 MiB 的整页对象全部静默失效。

> **服务端侧上限 `--max-msg`（v1.40+）**：默认 64 MiB，即"客户端不声明时给多少"。
> 它同时是本服务端接受的**上限**：客户端声明**高于**它会被**明确拒绝连接**并打日志，
> 而不是悄悄按小的开——后者会让客户端按自己声明的大小发包、打爆对端 recv buffer（RNR/QP 断）。
> 大集群建议显式设定，否则服务端的内存预算完全由客户端决定，而客户端常由别的团队部署、版本不一。

> ⚠️ 旧 `rail_affinity`（extra_config）**已废弃为 no-op**（v1.2.0）：它按 `tp_rank`
> 收窄选轨，但 DP-attention 下每 rank `tp_rank=0`→塌缩单轨。配了只打 stderr 告警。
> 用 `rdma_numa` / `DFKV_RDMA_NUMA` 替代。

### 1.3 wire 协议版本

当前 native TCP 与 RDMA 使用不同的显式 epoch：
- **TCP**：epoch 6，50-byte request prefix。
- **RDMA transport v2**：epoch 7；GET request 在相同 50-byte prefix 后携带
  目标 MR，payload 走 one-sided WRITE。

两种 prefix 都携带 64-bit tenant hash + 128-bit object digest。旧 epoch 直接
拒绝而不解码；client/server 必须按 [DEPLOY.md](DEPLOY.md) §4e 的隔离 ring
方式切换，不能依赖 rolling 混跑兼容。

### 1.4 原生 namespace、对象 key 与 raw value

`dfkv_open_v2(&options)` receives one immutable, size-delimited construction
descriptor. It contains either static members or MDS discovery settings, the
connector-produced binary namespace bytes, batch concurrency, and optional
client registration identity. Unknown flags/version/short structs fail closed.
There are no post-open membership mutators, operator namespace aliases, or
geometry parameters. The namespace binds the exact runtime model identity and
connector raw-layout ID (`sglang-hicache/raw-v1`, `vllm/raw-v1`,
`lmcache/raw-v1`).

所有 connector 的对象 key 都是 self-delimiting binary bytes，编码顺序为：

```
"DFKVPOOL\x02"
u32le(len(pool)) || pool
u32le(len(page_hash)) || page_hash
(u32le(size) || i32le(rank)) × [dp,tp,pcp,dcp,pp]
u32le(group)
u32le(len(component)) || component
```

`pool`、`page_hash`、`component` 可含 NUL、分隔符与非 UTF-8 字节；它们不做
文本 decode/re-encode，也不经过 Python `hash()`。SG key 在上述完整 bytes 后
追加 `"DFKVSG\x02" || u32le(width) || u32le(group)`。namespace 是另一段独立
binary identity，不拼入对象 key。namespace 与对象 key 经确定性、长度分帧的
SHA-256，
截取 128 bit 作为 object digest。native `BlockKey` 另带 64-bit tenant hash：
canonical `DFKVNS\0\2` 取第一个 length-framed tenant field；其它或 malformed
namespace 以完整 namespace bytes 为 tenant identity，再计算
`SHA256("DFKVTENANT1" || u64le(len) || identity)[:8]`（big-endian）。
同一 tenant 的不同 object hash 相同 tenant field，不同 tenant 隔离。
members/MDS、transport、telemetry、容量等是 control metadata，不参与 identity。
value 是调用方原始字节，实际存储长度单独返回；没有几何或 dtype 守卫。

clean v2 C ABI 对每个 scalar key 传 `(const void *key, uint64_t key_len)`；
batch/SG 传一一对齐的 `(const void *const *keys, const uint64_t *key_lens)`。
ctypes caller 持有每段 key buffer、pointer array 与 length array 直至 native
调用返回。没有 C string/NUL 终止语义，也没有旧 key ABI fallback。

`dfkv_register_memory(client, base, size)` 返回 `0` 才表示 MR 注册成功，native
transport/KVClient 的任何失败都返回非零。vLLM 把失败作为启动错误抛出；
LMCache 对显式 RDMA arena 同样 fail-fast；HiCache 的可选预注册路径记录警告并
保留该区域为未注册状态，绝不把失败计作已注册。

namespace 或 key 不同就是**冷 miss**。相同 namespace+key 却使用不同 dtype、
page/chunk size、shape、层顺序或内存布局是**operator error**，不会被 dfkv
改写成安全 miss。此类 identity-bearing 变化必须进入 model/schema namespace
或对象 key。新格式不读旧 key、不双写旧格式。

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
| `--store-engine=file\|slab` / `DFKV_STORE_ENGINE` | **server** | 该节点的盘上存储引擎；均未设置时默认 `slab`，`file` 仅为显式诊断回退 |
| `DFKV_RAM_TIER` / `DFKV_RAM_TIER_BYTES` | **server** | 写穿 RAM 热层 |
| `DFKV_SERVER_URING` | **server** | io_uring 异步 GET serve 路径 |
| `DFKV_SLAB_WRITE` | **server** | slab I/O 模式（默认 direct；`buffered` 为退出开关） |
| `DFKV_READ_COALESCE` / `_RECUR_MS` / `_TIMEOUT_MS` | **server** | v1.35 读侧 convoy 合并 + RAM 晋升（见根 README "Recommended tuning"） |
| `DFKV_TENANT_QUOTAS_FILE` / `DFKV_TENANT_DEFAULT_QUOTA_BYTES` | **server** | immutable per-node tenant capacity admission；客户端不要设置 |

显式配置的混合车队（部分节点 slab、部分诊断节点 file；部分带 RAM 层）对所有客户端**完全等价**。不配置 flag/env 的节点一律选择 slab；slab 配置无效时拒绝启动，不会自行加入为 file 节点。

受限 tenant 的超额 PUT 返回 `kQuotaExceeded`（上层看到普通 PUT failure，peer
不进入 I/O cooldown）；这与写入门/全盘压力的 `kCacheFull` 可由 server metrics
区分。quota 管理和 per-node sizing 见 [DEPLOY.md](DEPLOY.md) §3a。

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
# 数据面设备；多轨用逗号列表（标准节点 8×400G）
export DFKV_RDMA_DEV=ib7s400p0,ib7s400p1,ib7s400p2,ib7s400p3,ib7s400p4,ib7s400p5,ib7s400p6,ib7s400p7
export DFKV_RDMA_NUMA=1                   # 可选：多 NUMA 大机 NUMA 选轨（§1.2）
export DFKV_RDMA_MAX_PAYLOAD_BYTES=67108864  # 可选：单 chunk payload 上限，默认 64MiB
# DFKV_RDMA_DEPTH 可按容量分别设置；握手自动取两侧最小安全窗口
```

> ⚠️ hd04 当前只有 `ib7s400p0,ib7s400p1` 两轨 up，但标准训练计算网节点是 8×400G，
> 按本机实际 up 的口列全。

传输相关 env 亦可走 extra_config（构造 `dfkv_client_options_v2` 前解析，
extra_config 优先）：`"rdma_depth":K`、`"require_rdma":1`、`"rdma_numa":1`。

### 2.2 SGLang 启动 + 后端配置

**方案 A — MDS 动态发现（推荐）**：配 `mds_endpoints` + `mds_group`；插件把
发现参数放入 `dfkv_client_options_v2`，`dfkv_open_v2` 成功后自动轮询 MDS，
成员放置内容 epoch 变化时重建环，无需重启；epoch 不是 etcd revision。

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
    "mds_group":"default" }'
```

**方案 B — 静态成员表（遗留）**：无 MDS 时用 `members` 字段，节点增减需重启 SGLang：

```json
"members":"n57=192.168.1.57:28001,n58=192.168.1.58:28001"
```

（其余字段同方案 A，去掉 `mds_endpoints`/`mds_group`。）

**extra_config 全部键**（源 `dfkv_hicache.py`）：`interface_v1`（必填）、
`mds_endpoints`/`mds_group`（默认 `default`）/`mds_poll_ms`（3000）或 `members`、
`pcp_size`/`pcp_rank`、`dcp_size`/`dcp_rank`、`layer_num`（仅 L2-bypass 的 SG
分组控制，不进 namespace/value）、`lib_path`、`batch_concurrency`、
`rdma_depth`/`require_rdma`/`rdma_numa`、
`client_stats_poll_s`（10s，`0`=关）、
访问日志/telemetry 键（`access_log`、`access_log_path`、`metrics`、`tracing`、
`otlp_endpoint`、`trace_slow_request_ms`、`trace_sample_percent` 等，env 同义项见
§1.6）、`rail_affinity`（已废弃 no-op）。`model_name` 由 SGLang runtime 的
`HiCacheStorageConfig` 提供，不是 extra_config 键。

`pcp_size`/`dcp_size` 默认 `1`，此时对应 rank 固定为 `0`。任一 size
大于 `1` 时必须显式提供 `0 <= rank < size`；size/rank 不是整数、越界或缺失
都会在打开 dfkv client 前拒绝启动。PCP/DCP 是物理分片坐标，同一 page hash
在不同 rank 上生成不同 canonical object bytes，不能依赖默认 rank。

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
  `batch_set_v1/get_v1`；否则退回 generic `set/get`，而 dfkv 的 generic
  `get/batch_get` 是未实现的桩 → **写成功、L3 读静默失败**（线上踩过：
  launch 脚本漏配，14GB 写入但 prefetch 全 miss）。`interface_v1:1` 下 GET
  payload 经 RDMA 直落 HiCache 宿主页（client 零拷贝）；server O_DIRECT /
  io_uring 直读入注册 buffer，RDMA v2 以 one-sided WRITE 直落 client 目标
  MR，无 payload memcpy。
- MLA 下插件自动单对象、无 rank 后缀、`backup_skip`（仅 tp_rank0 写）。decode 共享前缀配同 members。
- **多池模型**（Mamba/SWA/DeepSeek-V4）用 v2 PoolTransfer 接口（插件已实现）。
  DSA/DeepSeekV4 主 `kv` 池是无数据的 LogicalHostPool（`get_page_buffer_meta→None`），
  插件对其 `batch_set_v1` 写空 marker 锚定命中前缀、`batch_get_v1` no-op，真实 KV 走 v2 侧池。
- **identity/layout 必须协同发布。** namespace 使用 SGLang runtime 给出的精确
  `model_name` + `sglang-hicache/raw-v1`；同一模型的 pool/hash/并行坐标/component
  进入 canonical object key。dfkv value 只有 raw bytes，不会检查 page size、
  dtype、shape 或层顺序。若这些布局在同一 `model_name` 下发生变化，先在代码中
  bump source-controlled raw-layout ID，再同时发布所有 writer/reader，并接受一次
  冷缓存。namespace/key 不一致只会 cold miss；相同 namespace+key 下布局不一致
  是 type-safety violation。
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

### 2.6 HiCache 特性边界

以下是 HiCache 的实际适用范围，不能照搬 vLLM 连接器的数据形状或旧 RDMA
路径经验：

- **stock host L2 不用 SG；L2-bypass 使用 SG。** host MLA 每页已是一个连续
  packed-latent 对象，无需合并。device-direct pool 是 layer-first，§2.7 会按
  有效 SGE 宽度拆成带 binary SG width/group 坐标的对象。
- **io_uring async GET（server 侧）。** 构建启用 `DFKV_WITH_URING` 时默认开，
  `DFKV_SERVER_URING=0` 才关闭；多连接场景实测 neutral，少连接深 pipeline
  约 +6%，失败自动回同步并有指标。
- **`DFKV_RDMA_DEPTH` — 两侧无需强制相等。** 握手取最小值，按共享 segment
  容量和连接 fan-out 分别配置即可。
- **HiCache 命中/吞吐/延迟与 client 注册指标**已在 v1.5.2+ 内，无需额外动作。

---

### 2.7 L2-bypass（L1↔L3 device 直连，绕过 host 池）

把 SGLang 的 L2（host pinned 池）从数据路径上摘掉，GPU 显存与 dfkv 之间直接 GPUDirect RDMA。
省掉一次 device↔host 拷贝和整个 host 池的驻留内存，代价是对象按 SGE
宽度分组，并在 canonical binary key 后追加 `DFKVSG\x02` 与两个 uint32-LE
坐标。宽度与组号都是 identity；不同宽度的客户端互相 cold miss，不会读到
另一种分组布局。

#### 客户端（SGLang 侧）

| env | 值 | 说明 |
|---|---|---|
| `SGLANG_HICACHE_L2_BYPASS` | `1` | 总开关。关闭即回到原版 host L2 路径 |
| `SGLANG_HICACHE_L2_BYPASS_DEDUP` | `1` | 同前缀并发 SG GET 去重：后到的请求 park 等待，不重复拉取 |
| `SGLANG_HICACHE_L2_BYPASS_FUSE_DRAFT` | `0` | draft（EAGLE）是否与目标层融进同一次 RDMA op。**默认关**——收益未经重复取样确认，单次测量不足以认领 |
| `DFKV_RDMA_MAX_BLOCK_BYTES` | 见 §1.2.1 | bypass 下块 = `dfkv_max_sg_segs() × page_size × 每token每层字节`（实际还受剩余层数限制）；先查询 runtime width，再按公式配置 |
| `DFKV_RDMA_IO_MS` | 默认 `10000` | v2 只建小 control QP + 租共享 slot；通常保持默认。 |

SGLang 启动侧需配合 `--hicache-mem-layout page_first_direct` 与
`--hicache-io-backend direct`，并在 extra_config 传真实 `layer_num`；它只决定
device segment 分组，不写入 namespace 或 raw value。

请求 bypass 后采用 fail-closed：构造期若 RDMA/SG put+get/`layer_num` 不满足
会拒绝启动；GPU pool（包括 sidecar/draft）的 region discovery 为空、抛异常，
或任一必需 MR 被 native registration 拒绝，注册立即失败并撤销
`supports_device_transfer()` 能力，不能静默回落并继续宣称 device-direct。
只有未请求 bypass 的可选注册和 host pool 预注册保留 best-effort。

#### 冷连接池

server 启动期会注册 process-wide receive segment；新 data QP 只创建小
control buffers 并租 `depth × slot` 的 offset，不会为每连接注册
`qd × block`。若首轮连接失败，检查 `dfkv_rdma_v2_ready`、receive segment
free bytes、每轨注册状态和 server 协商日志；不要用增大握手超时掩盖共享
segment 容量或注册失败。

#### 服务端

L2-bypass 不需要独立 server 协议，但两项决定 v2 容量：

- `DFKV_RDMA_RECV_SEGMENT_SIZE`：按 §1.2.1 的 peak live/pooled QP 公式；
  观察 `dfkv_rdma_recv_segment_free_bytes` 和 `dfkv_rdma_v2_ready`
- `DFKV_RDMA_MAX_BLOCK_BYTES` + `DFKV_RDMA_DEPTH`：共同决定每 QP 的 lease；
  声明要覆盖原版 L2 的较大整页对象，且总 lease 必须纳入共享 segment 预算。

#### 验证清单

```bash
# 1. bypass 真的开了（数自证日志，别只看 env）
nerdctl logs <容器> 2>&1 | grep -c 'L2-bypass ENABLED'
# 2. 声明真的到了服务端（这行只在客户端真声明时出现）
journalctl -u dfkv-server | grep 'rdma conn: protocol=v2 declared=.*shared-slot='
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
只有返回 `0` 才继续；注册失败会抛出启动错误，不会带着未注册指针进入流量。
每 chunk 的多层段经 **scatter-gather 批量 API** 合并成一个 dfkv key
（一次多-SGE RDMA / chunk，而非每层段一次），key/磁盘读数 ~20×↓。

`DfkvStoreConnector` **只支持 RDMA**：每个 vLLM engine 进程都必须设置
`DFKV_RDMA=1`。`dfkv_open_v2` 后连接器会在启动 poller、热配置和任何流量前
校验 native handle 报告的 transport；非 `rdma` handle 会立即关闭并报错。
GPU 设备指针路径没有 TCP 或 host-bounce fallback。

### 3.0 角色与前置条件

| 角色 | 要求 |
|---|---|
| **dfkv 存储节点** | NVMe SSD + 400G RDMA 网卡（IB/RoCE）；跑 `dfkv_server`（可选 `dfkv_mds`+etcd 动态成员） |
| **推理节点** | H100/A100 等 GPU + 同一 RDMA fabric；跑 vLLM（≥0.23.0） |
| **GPUDirect** | GPU 节点须加载 `nvidia-peermem`（`lsmod \| grep nvidia_peermem`），否则 `ibv_reg_mr` 拿不到 GPU MR |
| **KV 可再生** | dfkv 是纯 cache：节点丢失 = miss = 重算，无副本、无对象存储兜底 |

dfkv 与 vLLM 可同机（GPU 节点既跑 server 又跑 vLLM，池化本机 NVMe），也可分离。
把 `libdfkv.so` 拷到推理节点，然后在源码仓或 release tarball 解压根目录安装共享包和
vLLM 包（`dfkv-vllm` 精确依赖同包内的 `dfkv-common`）：
```bash
python -m pip install integration/common integration/vllm
```

### 3.1 启动 vLLM

**推荐（生产）：MDS 动态发现** —— 配 `mds_endpoints` + `mds_group`。连接器要求
**`mds_endpoints` 或 `members` 二选一**，设了 `mds_endpoints` 即优先走 MDS。

```bash
DFKV_RDMA=1 \
DFKV_RDMA_DEV=ib7s400p0,ib7s400p1,ib7s400p2,ib7s400p3,ib7s400p4,ib7s400p5,ib7s400p6,ib7s400p7 \
DFKV_LIB=/opt/dfkv/libdfkv.so \
vllm serve <model> \
  --tensor-parallel-size 2 --data-parallel-size 4 \
  --prefix-caching-hash-algo sha256 \
  --kv-transfer-config '{
    "kv_connector": "DfkvStoreConnector",
    "kv_connector_module_path": "dfkv_vllm.connector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "mds_endpoints": "192.168.0.8:28150,192.168.0.9:28150,192.168.0.10:28150",
      "mds_group": "glm",
      "batch_concurrency": "8"
    }
  }'
```

`model_name` 取 vLLM 的精确 `model_config.model`（即上面的 `<model>`），不是
extra-config 键。namespace 始终绑定该 model identity 与 `vllm/raw-v1`；没有可配置
的 namespace alias。

**备选（单节点/简单部署）：静态成员表** —— 无 MDS 时改用 `members`，节点增减需重启：

```json
"members": "n1=192.168.1.1:28001,n2=192.168.1.2:28001"
```

> ⚠️ **`members` 端口必须是 server 的 `--rdma-port`（RDMA QP bootstrap 监听口），
> 不是主 `--port`。** 指错则每个 RDMA `put` 失败 `rc=-1`。

> **必须使用 `--prefix-caching-hash-algo sha256`。** dfkv native 层会对完整
> namespace/object key 再做 SHA-256，但无法修复上游已经写进 object key 的进程本地
> Python hash。连接器在 scheduler/worker 启动时同时硬门禁：`builtin` 即拒绝，
> `PYTHONHASHSEED` 即使固定也不能替代内容定义的身份契约。

### 3.2 验证

1. **首轮（cold）**：发一个长 prompt，记 TTFT。
2. **重启 vLLM**（或换一个 DP 实例）后**发同一 prompt**：连接器工作则 vLLM 跳过 prefill
   （调度日志 `num_computed_tokens` 接近满、`WAITING_FOR_REMOTE_KVS`），TTFT 大幅下降，
   **输出与 cold 逐字一致**。
3. server 侧 `dfkvctl stat --all` 或 `/metrics` 看 get 命中、写入量。

不命中排查顺序：确认启动参数为 `--prefix-caching-hash-algo sha256` → MDS 可达
（或静态 `members` 端口是否 rdma-port）→ effective namespace 与 canonical
object-key 坐标是否一致（§1.4/§5）。
namespace/key 不一致是预期 cold miss。**空环 / MDS 不可达**可直接在 vLLM
`/metrics` 上看：`vllm:dfkv_client_ring_members==0`（写无处可去）或
`vllm:dfkv_client_mds_reachable==0`（[METRICS.md](METRICS.md) §3.5）。

### 3.3 环境变量（每个 vLLM 引擎进程）

通用传输/发现/观测 env 见 §1；vLLM 侧要点：

| env | 默认 | 推荐 | 说明 |
|---|---|---|---|
| `DFKV_RDMA` / `DFKV_RDMA_DEV` | **无；`DFKV_RDMA=1` 必填** | `1` / 全轨列表 | vLLM 设备指针连接器仅支持 GPUDirect RDMA；unset/TCP 在构造期关闭并拒绝，无 fallback。见 §1.2 |
| `DFKV_RDMA_DEPTH` | `1` | 保持 1 | depth-flat（§1.2） |
| `DFKV_RDMA_NUMA` | `0` | 多 NUMA 大机 `1` | §1.2 |
| `DFKV_LIB` / `DFKV_BUILD` | — | so 路径 | 被 extra_config `lib` 覆盖 |
| `DFKV_ACCESS_LOG_*` | 关 | 排查时开 | §1.6；vLLM 侧记 `batch_get_auto_sg`/`batch_put_sg`/`batch_exist`/`register_memory` |
| `DFKV_CLIENT_STATS_POLL_S` | `15` | 默认即可 | 后台轮询把环/MDS 健康镜像到 vLLM `/metrics`（`vllm:dfkv_client_ring_members`/`mds_reachable`/`mds_unreachable_polls_total`/`transport_info`）；`0`=关。见 [METRICS.md](METRICS.md) §3.5 |
| `DFKV_METRICS_ENABLED` / `DFKV_TRACING_ENABLED` | 关 | 按需 | §1.6；vLLM 连接器 telemetry **只认 env**（不读 extra_config） |

### 3.4 `kv_connector_extra_config`

`model_name` 由 vLLM runtime 提供，不是本表配置项。

| key | 默认 | 推荐 | 说明 |
|---|---|---|---|
| `mds_endpoints` | — | `ip:port,...`（dfkv_mds 层） | **生产首选**；设了即走 MDS 动态发现，省略 `members` |
| `mds_group` | `default` | 如 `glm` | MDS 成员组名，= `dfkv_server --group` |
| `mds_poll_ms` | `3000` | 默认即可 | MDS 轮询间隔（ms） |
| `members` | —（与 mds_endpoints 二选一） | `n=ip:rdma-port,...` | **端口 = server `--rdma-port`** |
| `lib` | env 兜底 | so 绝对路径 | |
| `batch_concurrency` | `8` | **大池可调高到 ≈ 节点数** | 跨节点 fan-out，**真正的吞吐杠杆**（depth 是平的） |
| `load_async` | `True` | 保持 True | 异步 load，走 `WAITING_FOR_REMOTE_KVS`、不占关键路径 |
| `transfer_queue_capacity` | `256` | 保持默认，按压测调 | 每个 worker、每个方向的排队上限（`1..65536`）。满队列时非阻塞拒绝新任务：save 立即释放 finish/free fence，load 标记失败并重算；非法值启动即失败。 |
| `enable_cross_layers_blocks` | `False` | 默认 False | 仅当引擎分页布局层内交错时开 |
| `lookup_rpc_port` | ipc 自动 | 一般不设 | rank0 前缀查询 RPC，仅 socket 名冲突时设 |

连接器实现 vLLM `shutdown()` 生命周期钩子：先停止接单并取消排队任务，再等待当前
native 操作完成、join 收发线程，最后仅关闭一次 native client。因而正常退出不依赖
daemon 线程或进程终止；过载和退出期间都不会静默留下永久占用的 KV block。

### 3.5 按场景的推荐配置

- **单实例 / 单 DP**：`--prefix-caching-hash-algo sha256` + `DFKV_RDMA=1` +
  `batch_concurrency=8` 默认，depth 保持 1。
- **多 DP / 多实例共享池**：所有实例使用 `--prefix-caching-hash-algo sha256`，
  并保持 effective namespace、canonical key 坐标和 raw payload layout 一致（§5）。
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
| 写成功但**读永不命中** | 未使用 `--prefix-caching-hash-algo sha256`（当前连接器会启动失败）；或 effective namespace / canonical object key 不一致（后两者表现为 cold miss） |
| 命中后输出/shape 错误 | 同一 namespace+key 被不同 dtype/page/shape/layout 复用；这是 type-safety violation。停写，bump source-controlled raw-layout ID 并同时发布所有 writer/reader |
| 每个 RDMA `put` 失败 `rc=-1` | `members` 指了 `--port` 而非 `--rdma-port` |
| `ibv_reg_mr` 失败 / 无 GPUDirect | GPU 节点没加载 `nvidia-peermem` |
| 首 token 偶发慢 ~2s | 每 DP rank 一次性 Triton JIT（非 bug）；预热可消 |
| 异构 HCA 的 SG 宽度不同 | binary SG 坐标把宽度纳入 key；不同宽度互相 cold miss。要共享就统一有效宽度 |

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
# 在源码仓或 release tarball 解压根目录执行；两个 sibling source root 必须一起保留。
python -m pip install integration/common integration/lmcache
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

**RDMA 版**：URL 用 **RDMA 端口**（server `--rdma-port`），并给 vLLM
进程加 `export DFKV_RDMA=1`。单 fabric 节点可直接使用自动 active-HCA 发现；
生产多 fabric 节点必须用 `DFKV_RDMA_DEV` 过滤出与 cache server 同 fabric
的本机设备（同型号节点可逗号列全轨，见 §1.2）。

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
      "model_name":"<exact-model-or-deployment-identity>"}}'

# 2) vLLM 指向 MP server（注意 --no-enable-prefix-caching 把全部 KV 复用交给 LMCache）：
vllm serve <model> --tensor-parallel-size 8 --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both",
    "kv_connector_extra_config":{"lmcache.mp.port":6555}}'
```

`adapter_params` 键：`url`（必填，语法同 in-process）、`membership`（`mds` 默认
或 `static`）、`lib`（否则 `DFKV_LIB`）、必填 `model_name`（MP-server 不会从
runtime metadata 自动提供）、`mds_poll_ms`（3000）、`num_workers`（8）、
`max_capacity_gb`（0 = 容量交给 dfkv 自管；>0 开 LMCache 聚合 L2 淘汰，见
§4.6.6）。MP-server API 不提供足以证明 byte-identical MLA replica 的 model/PP
元数据，因此对象 key 始终保留 `world_size/global_rank` identity，并拒绝手工折叠开关。
server 的 pinned L1 arena 在 LMCache 传入 `l1_memory_desc` 时自动注册 RDMA 零拷贝。

实现要点：dfkv 无原生 eventfd，`DfkvL2Adapter` 用**后台 asyncio loop + 三个
`create_event_notifier`（store/lookup/load）+ done-callback** 把同步 ctypes 客户端桥接到
`L2AdapterInterface`；`ObjectKey` 经统一 codec 渲染为 §1.4 的 canonical object key。
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
- `DfkvConnector` 把 `CacheEngineKey` 序列化成 binary object key，把 chunk 字节
  （`MemoryObj.byte_array`，LMCache 固定 host arena 的切片）经 `DfkvNativeClient`
  零拷贝交给 dfkv。
- dfkv `KVClient` 按 key 一致性哈希路由到 cache node，走 RDMA（或 TCP）读写。

#### 4.6.2 与 HiCache 插件的异同

| 维度 | HiCache（SGLang） | LMCache |
|---|---|---|
| 接口 | `HiCacheStorage`（batch_set_v1/get_v1…） | `RemoteConnector`（get/put/batched_\*…）/ L2 adapter |
| 底层 | `libdfkv.so`（ctypes）| 同 —— **相同** |
| key | canonical pool key（页 hash + 坐标/component） | canonical pool key（chunk hash + 坐标/component） |
| 块大小 | 固定页 | **任意**（`full_chunk_size_bytes`，可变长） |
| 零拷贝 buffer | SGLang host KV pool | LMCache `MixedMemoryAllocator` arena |

复用同一套 raw-value C ABI 与传输层；默认 namespace 含不同 layout ID，互不误读。

#### 4.6.3 相对 dingofs 版的两处实质改动

**① 任意块大小（含变长不满块）。** dingofs connector 把块硬编码 4 MiB（固定 io_uring
buffer）；dfkv 无此限制，直接存 LMCache 的 `full_chunk_size_bytes`（可几十 MiB）。
dfkv value 是 raw payload；权威存储长度保存在 store metadata 中并独立于 payload
返回。LMCache 的末个 chunk 可能不满，若用满块大小调用 fixed-size GET，会因
stored length 不等而 miss。变长 get 解决这个问题：
- C++：`KVClient::GetAuto(key, out, cap, *out_len)` /
  `BatchGetAuto(items, *out_lens)`；只要实际长度 `<= cap` 就返回 raw bytes 与长度。
- C ABI：`dfkv_get_auto` / `dfkv_batch_get_auto`。
- `BatchGetAuto` 复用 RDMA 零拷贝 `RangeInto` 路径：buffer 容量是上限，
  payload 直接散射进调用方 buffer，实际长度另行回传；connector 再调
  `reshape_partial_chunk(memory_obj, bytes_read)` 裁剪 shape。

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
    ├── key_mapper.py         # CacheEngineKey -> binary object key（完整 hash/ws/wid）
    ├── access_log.py         # 逐操作访问日志（默认关，DFKV_ACCESS_LOG_*）
    └── exists_cache.py       # ExistsLRU：「刚 put 又问存在」的远程往返短路
```

`native_client.py` 要点：
1. `load_lib(path)` 声明 v2 C ABI 的 `restype/argtypes`，包括
   `dfkv_open_v2`、raw-value get/batch/remove 和 memory registration；不探测或
   兼容 v1.x 构造符号。库路径优先级：显式 `lib` → `DFKV_LIB` →
   `$DFKV_BUILD/libdfkv.so` → dynamic-loader path。
2. 一个实例一个 handle：static/MDS 模式都在一次 `dfkv_open_v2` 中完成；
   对 `_collect_rdma_pools` 给出的区域调用 `dfkv_register_memory`，使 RDMA
   读写 host arena 切片无需逐操作 MR 注册；任何非零返回都会使启动失败，
   不会把 rejected arena 计作已注册。
3. 专用 `ThreadPoolExecutor(max_workers=get_parallelism)`，`loop.run_in_executor` 派发
   阻塞 ctypes 调用。`close()` 先停止接收新任务并等待已提交调用结束，再
   `dfkv_close`，避免 native handle 与在飞调用竞态。
4. 零拷贝指针：`(c_char*nbytes).from_buffer(mv)` 直接别名可写连续 buffer；只读 buffer
   退回 `from_buffer_copy`。keepalive 对象保活到 C 调用返回。
5. 返回结构：`batch_set→(ok, per_key)`；`batch_get→(ok, per_key, lengths)`；
   `batch_exists→per_key`。

`remote_connector.py` 要点：put 按 `len(byte_array)` 真实大小存（不满块也按真实大小写）；
get 分配满块 buffer、变长 get 拿回 `(per_key, lengths)`，命中后满块直接返回、不满块
`reshape_partial_chunk` 裁剪，非法长度当安全 miss 丢弃；`batched_get*` 保持 LMCache
「连续前缀」语义；缓冲分配走 `local_cpu_backend.allocate(...)`（host arena 切片）。

#### 4.6.5 namespace、object key 与 raw layout

in-process connector 从 LMCache runtime metadata 取精确 `model_name`；MP-server
路径要求 `adapter_params.model_name`。namespace 始终绑定该 identity 与
`lmcache/raw-v1`，不接受 operator alias。

in-process `RemoteConnector` 原样保留 LMCache runtime `CacheEngineKey` 的
`world_size/worker_id`。replicated MLA 识别、rank 归一化和 single-writer 选择均由
LMCache `RemoteBackend` 负责；dfkv 不再二次折叠或条纹分工，避免 PP/sharded 内容别名，
也避免 LMCache 已选单 writer 后 connector 再丢弃部分 chunk。

LMCache 的 `CacheEngineKey` / `ObjectKey` 统一编码为 §1.4 的
self-delimiting binary pool key。完整 hash、框架给出的 world size/rank、cache group
和 salt/component 都是 identity；字段按长度分帧，因此 NUL、分隔符与非 UTF-8
字节不会截断或别名。MP-server 路径始终保留 `world_size/global_rank` identity；
在上游 API 能表达可验证的 replicated/sharded 布局之前，不折叠 rank。
dtype、chunk size、shape 或序列化顺序若在同一 model identity 下变化，必须 bump
source-controlled `lmcache/raw-v1` layout ID；dfkv raw value 没有 geometry guard。

成员发现：URL `dfkv://<endpoint>/<group>` 由 `membership` 决定解释——
**mds（默认）**把 endpoint/group 写入 v2 options，由 constructor 启动后台
发现；**static** 把 endpoint 作为字面成员串，`<group>` 不使用。MDS 模式首次
refresh 前 ring 可能为空，早期操作安全 miss（LMCache 重算）。

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
  不满块、cap 过小（miss）、实际长度回传和双节点批量混合大小。
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

## 5. 共池、namespace/object key 与 raw layout（跨连接器）

**可以共用同一个 dfkv 集群/哈希环。** members/MDS group、transport 与容量是
control plane；不同模型和 runtime 共用节点与 LRU，不等于共用 cache identity。

默认自动 namespace 同时包含精确 runtime model identity 和 connector layout ID：
`sglang-hicache/raw-v1`、`vllm/raw-v1` 或 `lmcache/raw-v1`。因此同名模型在不同
runtime 默认也隔离。object key 再编码 pool、完整内容 hash、DP/TP/PCP/DCP/PP
坐标、cache group、component 和可选 binary SG 坐标。

**不一致的 namespace 或 object key = cold miss。** 这适用于 model identity、
parallel coordinates、component 或 scatter width 的任何差异。

**相同 namespace+key = 相同 raw payload layout 的强约定。** dfkv 不在 value 中
保存或校验 dtype、page/chunk size、shape、层顺序或 geometry。用相同 identity
写入不同布局是 type-safety violation，可能覆盖成可命中但不可解释的 bytes。
布局变化时必须 bump connector 的 source-controlled layout ID；跨 runtime 共享
只允许代码级、可审查的相同 namespace 与 canonical object-key schema，并须证明
payload bytes 完全兼容。

共池铁律：**control plane 可共享；identity 不同是冷缓存；identity 相同必须 byte-compatible。**

---

## 相关文档

- [DEPLOY.md](DEPLOY.md) — dfkv 集群（server + MDS）标准部署
- [ARCHITECTURE.md](ARCHITECTURE.md) — 存储/协议架构（wire 协议、slab、RAM 热层）
- [datapath-perf-notes.md](datapath-perf-notes.md) — RDMA 数据面性能笔记（depth-flat、SG coalescing）
- [access_log.md](access_log.md) / [METRICS.md](METRICS.md) / [tracing.md](tracing.md) — 可观测性
- `integration/vllm/README.md` · `integration/lmcache/README.md` — 包内快速参考
- `docs/superpowers/specs/2026-06-18-dfkv-vllm-store-connector-design.md` — vLLM 连接器设计文档
