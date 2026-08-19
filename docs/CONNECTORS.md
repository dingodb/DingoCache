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
| 零拷贝 | 两端零拷贝（GET 直落 HiCache 宿主页，**host-host**） | PUT 为 **GPUDirect RDMA**；GET 先落 retry-owned pageable staging，最终成功后经有界复用 pinned bounce pool 同步发布到 GPU | host-host 零拷贝（LMCache pinned arena 一次注册 MR） |
| 块大小 | 固定页（page_size token） | 变长 chunk（SG 多层段合并为一 key） | **任意**（含变长不满末块，走 `GetAuto`） |
| 典型场景 | SGLang PD 生产（GLM-5.1/5.2 MLA） | vLLM 生产直连（DeepSeek-V4-Flash 多池已验证） | vLLM+LMCache 栈；MP-server 路径给多 KV-group 模型 |

三者共用同一套 `libdfkv.so` C ABI、传输层与 MDS 发现，value 都是无 dfkv
信封的原始字节。自动 namespace 还绑定连接器的 raw-layout ID，默认互相隔离；
对象 key 统一使用 §1.4 的 canonical schema。三条路径都是**纯 Python 对接、
无需编译 CPython 扩展**。

**原生身份/裸值切换是 clean break。** 新 client、server 与 connector 应一起
升级并接受一次冷缓存；不会读取旧 key，也不会双写旧身份或旧 value 格式。

**同版本同时具备的服务端能力**（零客户端改动，对接方了解即可）——
① **服务端多轨 anchor**（`--rdma-dev` 逗号列表）：客户端按 §1.2 配轨亲和后即可吃满
8 轨；客户端与服务端应同版本整体升级。
② **读侧 convoy 合并 + RAM 晋升**（服务端 opt-in `DFKV_READ_COALESCE=1`，见根
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
| ~~`DFKV_MEMBERS`~~ | — | 仅测试脚本读取（如 `test/python/rdma_e2e_validate.py`），连接器不读；连接器静态成员走各自 config 的 `members` 键，生产优先 MDS 发现。 |

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
key 在 TTL（30s）内自动过期，无显式反注册、无脏 key。三条路径（SGLang HiCache /
vLLM / LMCache）行为一致，默认开，env `DFKV_CLIENT_REGISTER=0` 或连接器 config
`client_register=0` 关闭。SGLang HiCache 的注册信息串为 `type=hicache,model=<m>,tp_size=..,
tp_rank=..,ver=<lib>`（无 `role`——HiCache 是前缀 L3 缓存，无生产/消费角色之分）。观察：
`dfkvctl clients --mds <ep,...> --group <g>` 或 `dfkv_mds_group_clients` 指标。注意只有
**升级过的**客户端才注册，空列表 = "当前消费方都没注册" 而非 "没人用"。

### 1.2 传输（TCP / RDMA）

| env | 默认 | 推荐 | 说明 |
|-----|------|------|------|
| `DFKV_RDMA` | 一般路径未设 = TCP；**vLLM 直连无默认，必须 `1`** | 按连接器选择 | `1` 显式选择 native-verbs RDMA v2；请求 RDMA 后设备或协议不可用会失败，不会自动选择 TCP。`DfkvStoreConnector` 只接收 GPU 设备指针，构造时会关闭并拒绝任何非 RDMA handle。 |
| `DFKV_RDMA_DEV` | 首个 `ACTIVE` 本地 HCA | 留空让两端各自选本地首口；多轨才显式写同 fabric 白名单 | 留空时 bootstrap 不发送设备名，client/server 可使用不同本地命名。逗号列表显式开启多轨，新连接在健康轨间轮转；显式设备名会发给 peer，故两端必须存在同名且互通的 fabric。设备名上限 **18 字节**（v2 bootstrap dev frame 限制），超长即 fail-fast 拒绝启动/建连，不会静默截断（见 `src/transport/dev_frame.h`）。 |
| `DFKV_RDMA_DEPTH` | `4` | 两侧可不同，按容量选 | 握手协商 `min(client,server)` 作为安全窗口。每连接注册 `2 × depth × (18 B + 32 KiB)` 的有界 SEND/RECV control buffer；pull-read 连接还从共享 segment 各租 `depth` 个 receive slot 和 source slot。 |
| `DFKV_RDMA_MAX_BLOCK_BYTES` | 4 MiB 安全上限 | 按连接器块几何精确设置 | DCP2 声明本连接最大 PUT/GET block，决定共享 segment 的 slot 大小；超声明请求在客户端失败且不上 wire。声明越准，同一 segment 可容纳的 live/pooled v2 连接越多。 |
| `DFKV_RDMA_RECV_SEGMENT_SIZE` | 16 GiB | 按下文 live/pooled 连接公式设置；xb01 为 64 GiB | server 启动时申请，并在每个选中 rail 的共享 PD 上注册；失败会拒绝启动，segment 无可用 lease 时拒绝新连接。 |
| `DFKV_RDMA_NUMA` | `0` | 显式多轨的大机可设 `1` | 建连时按调用线程 NUMA 选本地 rail（无本地 rail→轮转白名单），server serve 线程跟随 QP rail。单块共享 receive segment 不做 per-rail NUMA 分配；仅保证选轨/线程亲和。 |
| `DFKV_RDMA_MAX_PAYLOAD_BYTES` | 64 MiB（67108864） | — | 客户端单 value payload 上限（不得超过 server 侧同名上限） |
| `DFKV_CUDA_PINNED_POOL_BYTES` | 64 MiB（67108864） | 按进程允许的 CUDA pinned host memory / memlock 设置 | vLLM CUDA GET publication 的进程级 bounce pool 预算。取正十进制字节数，最大 4 GiB；预算向下取整为完整 slot，且至少容纳一个 slot、最多 4096 个。非法或与 slot 不兼容的组合告警并回退整组默认值。slot 按需创建，因此实际 pinned high-water 不超过取整后的预算。 |
| `DFKV_CUDA_PINNED_SLOT_BYTES` | 4 MiB（4194304） | 保持能容纳常见单块；不必按最大 payload 预分配 | 固定 publication slot 大小，取 4 KiB–64 MiB 的正十进制字节数。非法值告警并回退默认 sizing。大 payload 按 slot 大小分块发布，不要求一个 payload 对应一个 slot。 |

**v2 数据面**：PUT 把 `[request prefix | raw payload]` 以
`RDMA_WRITE_WITH_IMM` 直接写入 server 租出的 slot；GET 先用 SEND 提交
`{addr,rkey,len}` 目标描述符，server 再以 RDMA WRITE 直接散射到调用方
buffer，最后只 SEND 状态与 authoritative stored length。两向 block payload
都不经过 control buffer。`kMembers` 在隔离的 control lane 上使用显式
`18-byte prefix + 32-KiB data` 容量；边界值完整返回，更大响应失败而不截断。

**CUDA GET publication**：RDMA 仍写入每次请求自有的 pageable retry
staging，而不是直接写调用方 CUDA 地址。这样失败 rail 的部分写入不会污染调用方
buffer；只有最终获胜的完整尝试才进入 publication。进程级 pool 对固定大小 slot
进行独占租赁；无空闲 slot 且预算已用尽时调用会阻塞等待（backpressure），不会
继续分配。每个 slot 首次按需创建时使用 portable CUDA pinned host allocation；
若该分配路径不可用，则以对齐 host allocation 加 portable CUDA host registration
作为回退。slot 在其生命周期内只分配并 pin/register 一次。publication 对每块先
从 pageable staging 复制到当次 leased slot，再在保留调用线程 CUDA context 的前提下
排入异步 H2D copy；该块 stream 同步完成后才归还 slot。大 payload 逐块重复
lease/copy/sync/release 流程。因此 CUDA GET 仍是 staged H2D，而不是 direct
GPUDirect GET：后者无法满足“失败重试不得改动 caller buffer”的 publication
fence。PUT 的 GPUDirect 路径不变。

容器和进程的 pinned-memory / `RLIMIT_MEMLOCK` 必须覆盖 pool **实际按需分配的
high-water**（上限是向 slot 取整后的 pool budget），还要为进程内其他 locked
memory 留余量。pool acquisition、pin/register、CUDA copy 或 stream sync 出错会
令 publication fail closed；sync 失败的 slot 会被永久隔离，并保持 pinned 直到
进程退出，避免释放 driver 仍可能引用的内存。正常 teardown 等待 active lease
归还后释放其余 pool slot 并记录 driver 释放错误；`fork()` 后的 child 拒绝使用
parent 继承的 CUDA pool 状态。

#### 1.2.1 `DFKV_RDMA_MAX_BLOCK_BYTES` 怎么定（含 L2 / L2-bypass 两套公式）

这个值在 **v2** 决定共享 receive segment 的 slot 大小：
`align4K(4 KiB + 声明的最大 raw payload)`。每条 pull-read 数据连接分别租
`depth` 个 receive slot 和 source slot；所有连接共享一块启动期注册的
`DFKV_RDMA_RECV_SEGMENT_SIZE`，因此声明保持精确仍有价值。

**共享 segment 容量必须按连接寿命算，不是按同时在飞请求算。** 数据 QP 的
slot 为

```
S_data = align4K(4096 + DFKV_RDMA_MAX_BLOCK_BYTES)
S_control = align4K(4096 + (18 + 32768)) = 40960
B_required >= 2 × (N_data × depth × S_data
                    + N_control × depth × S_control)
```

`N_data` / `N_control` 是该 server 上所有 rank、进程的**峰值 live + client
pool 中空闲连接**；lease 一直保留到 QP 被销毁或 `DFKV_RDMA_IDLE_MS` 回收，
线程峰值留下的 pooled QP 也要计入。4 MiB 声明、depth=4 时每条 pull data
QP 占 33,587,200 B；16 GiB 最多约 511 条、64 GiB 约 2046 条（均未扣
control lease）。上线同时观察
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
2. 起一轮真实负载，读服务端/客户端日志里的 `rdma: max block observed <N>B` 高水位
3. 取实测值的 2~4 倍设定，注意**必须同时覆盖原版 L2 路径的整页对象**
4. 复核服务端日志 `rdma conn: protocol=v2 declared=… control=… shared-slot=… qd=…` 确认生效

🔴 **设小后上层仍只看到 miss——这是最危险的部分。** 超声明的块被判 `kInvalid`，
而 `kInvalid` 被客户端健康计数刻意忽略，上层 `hits[i] != 1` 与“这页压根没缓存”
无法区分：不崩、不熔断。因此会打 `rdma: block …B exceeds the declared bound
…B` 告警（首次 + 每 1024 次），所以必须纳入日志告警。
典型踩法：照 L2-bypass 实测的 1.02 MiB 调到 2 MiB，切回原版 L2 后
2.74 MiB 的整页对象全部静默失效。

> **服务端侧上限 `--max-msg`**：默认 32 MiB，即"客户端不声明时给多少"。
> 它同时是本服务端接受的**上限**：客户端声明**高于**它会被**明确拒绝连接**并打日志，
> 而不是悄悄按小的开——后者会让客户端按自己声明的大小发包、打爆对端 recv buffer（RNR/QP 断）。
> 大集群建议显式设定，否则服务端的内存预算完全由客户端决定，而客户端常由别的团队部署、版本不一。

> `rail_affinity=true` 在 native client 创建前把完整 `DFKV_RDMA_DEV` 收窄为
> 一条 primary 和默认相邻 1 条 fallback。SGLang HiCache 使用其物理 attention
> coordinate（PCP>1 时 `pcp_rank`，否则 `tp_rank`）；vLLM 使用 world-group
> per-host `local_rank`，不把可重复的 TP/DCP/PCP/PP 存储坐标误当 HCA 序号。
> `rail_affinity_fallbacks=0` 恢复严格 one-rank/one-rail；connector 设置
> `DFKV_RDMA_PRIMARY_DEV=<primary>` 和 `DFKV_RDMA_NUMA=0`。

### 1.3 wire 协议版本

当前 native TCP 与 RDMA 使用不同的显式 epoch：
- **TCP**：epoch 6，50-byte request prefix。
- **RDMA transport v2**：epoch 7；GET request 在相同 50-byte prefix 后携带
  目标 MR，payload 走 one-sided WRITE。

两种 prefix 都携带 64-bit tenant hash + 128-bit object digest。旧 epoch 直接
拒绝而不解码；client/server 必须按 [DEPLOY.md](DEPLOY.md) §4e 的隔离 ring
方式切换，不能依赖 rolling 混跑兼容。唯一例外是 MDS 控制面：
`DFKV_MDS_ACCEPT_LEGACY=1` 可让 v2 MDS 以旧 epoch 帧服务 v1.x 节点/客户端的
注册与心跳（按 epoch 判别，status 按 v1 枚举顺序回声），**数据面仍严格拒绝**——
v1/v2 须分属不同 group，见 [DEPLOY.md](DEPLOY.md) §2b。

### 1.4 原生 namespace、对象 key 与 raw value

`dfkv_open_v2(&options)` receives one immutable, size-delimited construction
descriptor. It contains either static members or MDS discovery settings, the
connector-produced binary namespace bytes, batch concurrency, and optional
client registration identity. Unknown flags/version/short structs fail closed.
There are no post-open membership mutators, operator namespace aliases, or
geometry parameters. The namespace binds the exact runtime model identity and
connector raw-layout ID (`sglang-hicache/raw-v1`, `vllm/raw-v1`,
`lmcache/raw-v1`). 可调字段只有 `tenant_id`（默认 `default`）与
`model_revision`（默认取 model identity），走各连接器的 extra_config/config 键
（§2.2 / §3.4 / §4.5）——多租户隔离或模型版本滚动时显式分开
namespace；operator namespace alias 一律被拒绝（fail-closed）。

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
| `DFKV_TP_RANK` | — | 仅 vLLM connector：用作 `connector_id` 遥测 label 的 rank 后缀；写复制/缓存身份由 canonical key 决定，不受它影响 |
| `DFKV_CLIENT_NODE_DEDUP` | `0` | 客户端侧节点内 rendezvous 去重：同一引擎进程内**并发去同一目标节点的相同 batch 操作合并领头**，免重复 wire 往返。replicated-MLA 场景（vLLM / HiCache）自动置 `1`，显式 `0` 可关 |
| `DFKV_CLIENT_NODE_DEDUP_GPU` | `0` | 上一条的 GPU（device-pointer）路径开关；vLLM replicated-MLA 在 host dedup 开启且未显式设置时自动 `1` |
| `DFKV_READ_SHARD_KEYS` | `16` | 每读分片的目标 key 数：把单节点的一组批量 GET 切成多分片并发（少节点环/大 batch 集中单节点时突破 ~166 MB/s/conn 的单连接串行 drain 天花板；宽环上无感） |
| `DFKV_READ_MAX_CONNS` | `8` | 单节点读分片的并发连接上限（与上一条配对；`1` = 关闭分片） |
| `DFKV_FANOUT_THREADS` | `32` | 客户端批量操作 fan-out 线程池上限（clamp [1,1024]）；高并发引擎（callers × node-groups ≫ 32）不调会退化 caller-serial，per-call 延迟从 max(group) 变 sum(group) |

以上三条为 **native C 客户端**（`libdfkv.so`）knob，对 HiCache / LMCache / vLLM
三条接入路径同等生效；生产漂移排查时先看启动 config dump（全 knob 带来源打印）。

### 1.6 可观测性（opt-in，全部不占数据路径）

三条路径共用同一套观测设施；差别只在**配置入口**：HiCache 插件支持 extra_config 键
**或** env（extra_config 优先），vLLM / LMCache 连接器的 telemetry **只认 env**。

| 层 | 打开方式 | 详见 |
|----|---------|------|
| **逐操作访问日志**（一行一 op：`<op>(<args>) : <result> <秒>`，如 `batch_get_auto_sg(20 keys) : hits=20/20, 1310720 bytes <0.007234>`；关 ≈100ns 空操作，开 = 异步落盘、热路径 ~µs） | `DFKV_ACCESS_LOG_ENABLED=1`、`DFKV_ACCESS_LOG_PATH`（空=stderr）、`DFKV_ACCESS_LOG_THRESHOLD_US`（只记 ≥N µs 的 op，0=全记）、`DFKV_ACCESS_LOG_MAX_BYTES`/`_BACKUP_COUNT`（滚动） | [access_log.md](access_log.md) |
| **车队指标 push（OTLP→Collector→Grafana）**：命中率/吞吐/op 延迟 + 逐 peer 延迟 | `DFKV_METRICS_ENABLED=1` + `OTEL_EXPORTER_OTLP_ENDPOINT=http://<collector>:4318`（默认 stdlib OTLP/HTTP JSON；`:4317` 只给显式 OTel SDK gRPC exporter）；可选 `DFKV_METRICS_EXPORTER`（`stdlib` 默认零依赖 / `otel`）、`DFKV_METRICS_EXPORT_INTERVAL_MS`（10000）、`DFKV_PROBE_INTERVAL_MS`（空闲也出逐 peer 延迟）、`DFKV_CLIENT_STATS_POLL_S`、`DFKV_PEER_LATENCY_POLL_S` | [METRICS.md](METRICS.md) §3.4、[deploy/observability/CONNECTOR-USAGE.md](../deploy/observability/CONNECTOR-USAGE.md) |
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
| `DFKV_READ_COALESCE` / `_RECUR_MS` / `_TIMEOUT_MS` | **server** | 读侧 convoy 合并 + RAM 晋升（见根 README "Recommended tuning"） |
| `DFKV_TENANT_QUOTAS_FILE` / `DFKV_TENANT_DEFAULT_QUOTA_BYTES` | **server** | immutable per-node tenant capacity admission；客户端不要设置 |
| `DFKV_TCP_FIRST_REQ_MS` / `DFKV_MDS_FIRST_REQ_MS` / `DFKV_METRICS_FIRST_REQ_MS` | **server / mds** | 各 listener 首请求 deadline（默认 30000ms，`0`=关）：deadline 内不发首个请求的连接被踢，防空连接堆积 |
| `DFKV_METRICS_MAX_CONNS` / `DFKV_MDS_MAX_CONNS` | **mds / metrics** | listener 并发连接上限（metrics 64、MDS 4096），防连接洪泛 |

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
`rdma_depth`/`require_rdma`/`rdma_numa`、`rail_affinity`/
`rail_affinity_fallbacks`，canonical namespace 的可配键 `tenant_id`
（默认 `default`）、`model_revision`（默认取 model identity），以及
identity/layout 字段 `page_size`（默认 64）/`kv_cache_dtype`/`dtype_tag`/
`head_num`/`head_dim`/`dp_size`——这些都进 canonical namespace，改动即换
namespace（表现为一次冷缓存）；
`node_dedup`（对应 env `DFKV_CLIENT_NODE_DEDUP`，见 §1.5）、
`backup_exist_gate`（save 前的 exist 探测门）、
`client_stats_poll_s`（10s，`0`=关）、
访问日志/telemetry 键（`access_log`、`access_log_path`、`metrics`、`tracing`、
`otlp_endpoint`、`trace_slow_request_ms`、`trace_sample_percent` 等，env 同义项见
§1.6）。`model_name` 由 SGLang runtime 的 `HiCacheStorageConfig` 提供，
不是 extra_config 键。

`pcp_size`/`dcp_size` 默认 `1`，此时对应 rank 固定为 `0`。任一 size
大于 `1` 时必须显式提供 `0 <= rank < size`；size/rank 不是整数、越界或缺失
都会在打开 dfkv client 前拒绝启动。PCP/DCP 是物理分片坐标，同一 page hash

#### 多模型/多租户配置（对应 §5 四维度）

分组按最关心配置写：

| 维度 | 配置位点 | 示例 |
|---|---|---|
| ① Group | `mds_group` extra_config | `"mds_group":"glm-prod"` |
| ③ tenant_id | `tenant_id` extra_config | `"tenant_id":"prod-chat"` |
| ④ model_name | SGLang `--served-model-name`（或 `--model-path` 解析而来） | `--served-model-name glm-5.2` |
| ④ model_revision | `model_revision` extra_config（默认 = model_name） | `"model_revision":"nvfp4-2026-08"` |

生产双业务线共环配置示例（同模型一权、两 tenant）：

```bash
sglang serve /models/glm-5.2-nvfp4 --served-model-name glm-5.2 \
  ... \
  --hicache-storage-backend-extra-config '{
    "backend_name":"dfkv","module_path":"dfkv_hicache","class_name":"DfkvHiCache",
    "interface_v1":1,
    "mds_endpoints":"10.201.3.10:28150,10.201.3.11:28150",
    "mds_group":"glm-prod",
    "tenant_id":"prod-chat",
    "model_revision":"nvfp4-2026-08"}'
```

布局参数变化（dtype/量化/层数/TP）**必须 bump `model_revision`**，
同 model_name+ 默认 revision = 自动隐含同 layout，可能命中 byte-incompatible bytes。
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
  `batch_set_v1/get_v1`；否则退回 generic `set/get`。dfkv 的 generic 路径
  已实现，但相比 `interface_v1` 多一次 host 中间拷贝；且 MLA 模型下 generic
  路径每个 rank 都会重复写同一页（无 `backup_skip` 单写者判定）。强制
  `interface_v1` 是为了**拒绝静默降级**——早期版本 generic 还是未实现的桩
  （线上踩过：launch 脚本漏配，14GB 写入但 prefetch 全 miss）；如今虽已
  实现，漏配仍会静默带来多一次拷贝与 MLA 重复写，插件仍 fail fast。
  `interface_v1:1` 下 GET
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
- **block hash 是 SGLang C++ native SHA256，无需 `PYTHONHASHSEED`。** SGLang
  `radix_cache.hash_page()` → `get_native_hash()` → `hash_binding.cpp` 的 C++
  SHA256 实现，`page_hash` 传入 dfkv 时已是确定性 hex string，不经过 Python
  `hash()`。跨进程/跨实例 block hash 天然一致，`PYTHONHASHSEED=0` 无需设置（设了
  无害但不参与身份契约）。这与 vLLM 连接器（§3.1）的硬门禁不同——HiCache 没有
  `builtin` 选项，SGLang 的 page hash 始终是 SHA256。v1 时代部分部署沿用的
  `PYTHONHASHSEED=0` 是 vLLM `builtin` 算法的历史遗留，HiCache 路径从未需要。
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
- **HiCache 命中/吞吐/延迟与 client 注册指标**已内置，无需额外动作。
- **client keepalive 默认开启。** `DFKV_RDMA_KEEPALIVE_MS` 默认 `15000`，每
  15 秒探测 idle pool 中的 QP，低于频繁重建场景推荐的 server
  `DFKV_RDMA_IDLE_MS=30000`。进程退出后探测自然停止，server 仍可在 30 秒后
  回收 receive-segment lease。keepalive interval 必须严格小于 server idle
  interval；不需要保活时显式设置 `DFKV_RDMA_KEEPALIVE_MS=0`。

#### 0064 B200 + GLM-5.2-NVFP4 实测（2026-08-09）

95,000-token 请求，8 个 attention-CP rank、8 条物理 rail、完整 L3 命中：

- API 返回 `cached_tokens=94,976`，server 读取 42.40 GB，RDMA completion
  error 增量为 0。
- 插件自身指标每 rank：主 KV 4.267 GB / 0.13–0.18 s，side pool
  0.978 GB / 0.034–0.041 s。dfkv 数据路径已低于 0.22 s/rank；API 端到端
  波动主要来自 SGLang 的恢复 prefill、rank barrier/JIT，而不是存储拉取。
- `DFKV_READ_MAX_CONNS=16` 在 8 rank 单机上形成最多 128 条并发 data QP，
  端到端反而从 3.61 s 退化到 44.94 s；保持默认 8。吞吐靠 rank-to-rail
  隔离和默认 8-way fanout，不能无限加连接。
- 不要为 attention-CP rank 打开 native
  `DFKV_CLIENT_NODE_DEDUP_GPU`：各 rank 的 page hash 可以相同，但目标 KV
  payload 是 rank-local shard。实测打开后仅恢复 32,768/94,976 tokens，
  请求退化至 61–87 s。SGLang 并发请求去重只使用
  `SGLANG_HICACHE_L2_BYPASS_DEDUP` 的框架级语义。

#### 0064 标准 L2 idle-tail 修复（2026-08-10）

server `DFKV_RDMA_IDLE_MS=30000`、client 未开 keepalive 时，45 秒空闲后的
4×100K-token L3 GET 每 rank 产生 12–17 次
`dfkv_rdma_client_stale_pool_retries_total`，单次 `batch_get` 最慢
3.779 s，TTFT 均值 18.313 s、吞吐 20,989 tok/s；completion timeout 与 rail
error 均为 0，排除了 SSD/IB 带宽不足。

client 设置 `DFKV_RDMA_KEEPALIVE_MS=10000` 后，同样 45 秒空闲窗口内 keepalive
保住所有 live QP；随后 GET 的 stale retry/new connection 增量均为 0，
`batch_get` 最慢 78.992 ms、TTFT 均值 4.595 s、吞吐 65,008 tok/s。对应
Prometheus 指标为 `dfkv_rdma_client_keepalive_{attempts,successes,failures}_total`
（connector pull 端附加 `tp_rank`，OTLP 中映射为
`dfkv_connector_rdma_keepalive_*_total`）。

该配置下继续增加 client/server fanout 没有存储侧收益；优化重点应转向
SGLang engine 的恢复 prefill 和跨 rank 同步。


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
# 3. 块大小与余量
nerdctl logs <容器> 2>&1 | grep 'max block observed'
nerdctl logs <容器> 2>&1 | grep -c 'exceeds the declared bound'   # 必须为 0
# 4. 协商后的深度窗口（服务端 rdma 协商日志，qd= 为握手取 min(client,server) 的真值）
journalctl -u dfkv-server | grep 'rdma conn: protocol=v2 declared=' | grep -o 'qd=[0-9]*'
```

## 3. vLLM 直连 — DfkvStoreConnector

`DfkvStoreConnector` 是 vLLM `KVConnectorBase_V1` 直连连接器：把 KV cache 接入
dfkv 集群，**绕开 LMCache**，占据与 `MooncakeStoreConnector` 相同的
`--kv-transfer-config` 槽位。生产者和消费者读写同一共享池，实现跨请求、跨实例、
跨重启的前缀复用。

连接器纯 Python（ctypes over `libdfkv.so`），接收 **GPU 设备指针**。分页 KV cache
经 `dfkv_register_memory` 一次注册（nvidia-peermem 下 `ibv_reg_mr` 产出 GPUDirect MR），
只有返回 `0` 才继续；注册失败会抛出启动错误，不会带着未注册指针进入流量。
PUT 保持从 GPU MR 发起的 GPUDirect RDMA。v2.19 的 GET 则由 native range API
显式标记 CUDA destination：RDMA 先写入该次操作拥有的 pinned host staging，只有
最终健康 rail attempt 成功后才用 pinned async H2D copy 发布到 GPU，并同步 CUDA
stream 后返回成功。失败 attempt 不会提前发布；连续 range、mixed host/CUDA SG 和
rail retry 使用同一 fence。

每 chunk 的多层段经 **scatter-gather 批量 API** 合并成一个 dfkv key
（一次多-SGE 操作 / chunk，而非每层段一次），key/磁盘读数 ~20×↓。

`DfkvStoreConnector` **只支持 RDMA**：每个 vLLM engine 进程都必须设置
`DFKV_RDMA=1`。`dfkv_open_v2` 后连接器会在启动 poller、热配置和任何流量前
校验 native handle 报告的 transport；非 `rdma` handle 会立即关闭并报错。
GPU PUT 没有 TCP fallback；CUDA GET 的 host staging 是 RDMA retry/publication
机制，不是 TCP fallback。

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
PYTHONHASHSEED=0 \
DFKV_RDMA=1 \
DFKV_RDMA_DEV=ib7s400p0,ib7s400p1,ib7s400p2,ib7s400p3,ib7s400p4,ib7s400p5,ib7s400p6,ib7s400p7 \
DFKV_RDMA_DEPTH=1 \
DFKV_RDMA_NUMA=0 \
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
      "batch_concurrency": "8",
      "rail_affinity": true,
      "rail_affinity_fallbacks": 1
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

> **必须同时使用 `--prefix-caching-hash-algo sha256` 和固定
> `PYTHONHASHSEED`。** 当前 vLLM 即使选择 SHA-256，未设置 seed 时仍以
> `os.urandom()` 初始化首块 parent hash，导致引擎每次重启后所有旧 key 静默
> cold miss。连接器对两项均有启动硬门禁。共享同一 store 的所有 producer 和
> consumer 必须使用同一个固定值，推荐 `PYTHONHASHSEED=0`。

### 3.1.1 正式 server / launcher 必须固化的参数

不要只在交互 shell 临时 `export`。这些参数必须写入实际创建 vLLM engine
进程的 systemd unit、Kubernetes Pod spec 或容器 launcher；仅更新 env 文件但
没有 `export`/`EnvironmentFile=` 不生效。

| 层级 | 必须固化 | 原因 |
|---|---|---|
| engine 参数 | `--prefix-caching-hash-algo sha256` | object key 必须是内容定义 hash；`builtin` 被连接器拒绝 |
| engine 环境 | `PYTHONHASHSEED=0`（所有共享实例同值） | 稳定首块 parent hash；缺失会在每次进程重启后全量 cold miss |
| engine 环境 | `DFKV_RDMA=1`、`DFKV_RDMA_DEV=<本机 ACTIVE HCA 有序全轨列表>` | 强制 RDMA；affinity 开启时 connector 在每个 worker 进程中按 world-group local rank 收窄为 primary + fallback |
| connector extra-config | `rail_affinity=true`、`rail_affinity_fallbacks=1` | bounded failover；native open 前自动设置 per-process `DFKV_RDMA_PRIMARY_DEV` 和 `DFKV_RDMA_NUMA=0` |
| engine 环境 | `DFKV_RDMA_DEPTH=1` | 每条 QP 限制一个在途 range；吞吐靠连接 fanout，不靠同 QP depth |
| 容器资源 | Docker/nerdctl `--ulimit memlock=-1:-1`；Kubernetes `ulimits` 等价配置；systemd `LimitMEMLOCK=infinity` | 允许 GPU MR 及 CUDA GET 临时 pinned host staging；按并发 GET payload 验证 pinned-memory 与 memlock 容量，8 MiB 默认值可能触发 `ibv_reg_mr` 失败或临时注册退化 |
| 宿主机 | `nvidia-peermem` 已加载 | GPUDirect MR 注册前提 |

nerdctl/Docker 参考（参数必须出现在**创建容器**时，容器启动后再执行
`ulimit` 无效）：

```bash
nerdctl run --ulimit memlock=-1:-1 \
  -e PYTHONHASHSEED=0 \
  -e DFKV_RDMA=1 \
  -e DFKV_RDMA_DEV=ib7s400p0,ib7s400p1 \
  -e DFKV_RDMA_DEPTH=1 \
  -e DFKV_RDMA_NUMA=0 \
  ... <image> \
  vllm serve <model> --prefix-caching-hash-algo sha256 ...
```

启动后验收：

```bash
grep 'Max locked memory' /proc/<engine-pid>/limits   # unlimited
tr '\0' '\n' </proc/<engine-pid>/environ |
  grep -E '^(PYTHONHASHSEED|DFKV_RDMA|DFKV_RDMA_DEV|DFKV_RDMA_DEPTH|DFKV_RDMA_NUMA)='
lsmod | grep nvidia_peermem
```

功能验收不能只测同进程第二次请求：必须写入长 prompt，**重启整个 engine
进程**后重复相同 prompt，确认 key 存在、cached tokens 接近完整 prompt，
且 `dfkv_rdma_completion_errors_total` 无增量。

### 3.2 验证

1. **首轮（cold）**：发一个长 prompt，记 TTFT。
2. **重启 vLLM**（或换一个 DP 实例）后**发同一 prompt**：连接器工作则 vLLM 跳过 prefill
   （调度日志 `num_computed_tokens` 接近满、`WAITING_FOR_REMOTE_KVS`），TTFT 大幅下降，
   **输出与 cold 逐字一致**。
3. server 侧 `dfkvctl stat --all` 或 `/metrics` 看 get 命中、写入量。

不命中排查顺序：确认 `--prefix-caching-hash-algo sha256` 与
`PYTHONHASHSEED` 同时存在且所有实例同值 → MDS 可达（或静态 `members`
端口是否 rdma-port）→ effective namespace 与 canonical object-key 坐标是否一致。
namespace/key 不一致是预期 cold miss。**空环 / MDS 不可达**可直接在 vLLM
`/metrics` 上看：`vllm:dfkv_client_ring_members==0`（写无处可去）或
`vllm:dfkv_client_mds_reachable==0`（[METRICS.md](METRICS.md) §3.5）。

### 3.3 环境变量（每个 vLLM 引擎进程）

通用传输/发现/观测 env 见 §1；vLLM 侧要点：

| env | 默认 | 推荐 | 说明 |
|---|---|---|---|
| `DFKV_RDMA` / `DFKV_RDMA_DEV` | **无；`DFKV_RDMA=1` 必填** | `1` / 本机有序全轨列表 | vLLM GPU 指针连接器仅支持 RDMA；`rail_affinity=true` 时 connector 在每个 worker 内收窄列表 |
| `DFKV_RDMA_DEPTH` | `4` | 保持生产已验证值 | depth-flat（§1.2） |
| `DFKV_RDMA_NUMA` | `0` | affinity 开启时保持 `0` | connector 为严格 primary/fallback 映射显式设置 0；affinity 关闭时才按需启用 NUMA 动态选轨 |
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
| `rail_affinity` | `False` | 多 rank、多 rail 生产设 `true` | 按 vLLM world-group per-host `local_rank` 选择 primary；在 native client 创建前设置每进程独立 rail 环境 |
| `rail_affinity_fallbacks` | `1` | `1` | 相邻有序 fallback 数；`0`=严格单 rail，超出可用 rail 数时自动收敛 |
| `load_async` | `True` | 保持 True | 异步 load，走 `WAITING_FOR_REMOTE_KVS`、不占关键路径 |
| `transfer_queue_capacity` | `256` | 保持默认，按压测调 | 每个 worker、每个方向的排队上限（`1..65536`）。满队列时非阻塞拒绝新任务：save 立即释放 finish/free fence，load 标记失败并重算；非法值启动即失败。 |
| `enable_cross_layers_blocks` | `False` | 默认 False | 仅当引擎分页布局层内交错时开 |
| `lookup_rpc_port` | ipc 自动 | 一般不设 | rank0 前缀查询 RPC，仅 socket 名冲突时设 |
| `client_register` | `1`（MDS 发现时） | 默认即可 | MDS 客户端注册开关（`0` 关；env `DFKV_CLIENT_REGISTER=0` 等价，见 §1.1） |
| `tenant_id` | `default` | 多租户时显式设 | canonical namespace 的 tenant 字段（§1.4） |
| `model_revision` | model identity | 模型版本滚动时显式设 | canonical namespace 的模型版本字段（§1.4）；改动即换 namespace（冷缓存） |

DCP（decode context parallel）宽度/Rank **不是**本表键：连接器从 vLLM 运行时取
（`get_dcp_group().world_size`），与 `tp_rank=-1`（replicated-MLA 存储坐标）一起进入
canonical key metadata（§1.4），勿在 extra_config 手工设定。

连接器实现 vLLM `shutdown()` 生命周期钩子：先停止接单并取消排队任务，再等待当前
native 操作完成、join 收发线程，最后仅关闭一次 native client。因而正常退出不依赖
daemon 线程或进程终止；过载和退出期间都不会静默留下永久占用的 KV block。

**MLA / SG 坐标语义（v2.0.0 修正点）**：
- **replicated-MLA**（MLA + TP>1 + DCP≤1）：每次 SAVE 都挂在单一 canonical
  存储坐标 `tp_rank=-1`；lookup dedup 探测也按 `tp_rank=-1` 原样探测。
  早期版本按 `tp_rank=0..tp_count-1` 展开探测，永远对不上 `-1` 坐标，
  replicated-MLA 外部 L3 命中恒为 0——升级到 v2.0.0 连接器前勿在 MLA 生产
  上预期外部命中。
- **多 kv_cache_group / SG 分组**：dedup/lookup 探测 chunk 的**全部**显式
  scatter group（不只是 group 0）；任一组缺失（partial write 或被淘汰）即判
  未缓存并整 chunk 重写，避免半存 chunk 当命中。
- **partial save 清理**：PUT 部分失败时，连接器 best-effort `batch_remove`
  清掉失败 chunk 的各兄弟 group key（`_remove_partial_sg_groups`，不抛、
  无 remove RPC 时直接跳过），防止半存 chunk 占环容量直到 eviction。

### 3.5 按场景的推荐配置

- **多模型 / 多租户（生产）**：四维度必须按 §5 显式在 `kv_connector_extra_config` 里设定；
  **布局参数变化必 bump `model_revision`**。
  ```json
  {
    "kv_connector":"DfkvStoreConnector",
    "kv_role":"kv_both",
    "kv_connector_extra_config":{
      "mds_endpoints":"10.201.3.10:28150,10.201.3.11:28150",
      "mds_group":"glm-prod",
      "tenant_id":"prod-chat",
      "model_revision":"nvfp4-2026-08"
    }
  }
  ```
  位点映射：`mds_group`（① 环）、`tenant_id`（③）、`model_revision`（④）。
  `model_name`（④）由 **vLLM 启动 `--model`/`--served-model-name`** 提供，不是 extra_config
  键；引擎/布局身份全量进 canonical namespace（§1.4）。
- **单实例 / 单 DP**：`--prefix-caching-hash-algo sha256` +
  `PYTHONHASHSEED=0` + `DFKV_RDMA=1`；`batch_concurrency=8` 默认，
  `DFKV_RDMA_DEPTH=1`。
- **多 DP / 多实例共享池**：所有实例使用
  `--prefix-caching-hash-algo sha256` 和相同固定 `PYTHONHASHSEED`，并保持
  effective namespace、canonical key 坐标和 raw payload layout 一致（§5）。
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
- **depth 平**：裸 GET 单连接 depth 1 = depth 32 ≈ 1.24 GB/s，完全一样（默认已改为 4）。
- **传输层**：裸 GET 8 连接 5.2 GB/s、16 连接 6.2 GB/s（详见 [datapath-perf-notes.md](datapath-perf-notes.md)）。

0064 B200 + GLM-5.2-NVFP4（94,000-token，2026-08-09）补充：

- 固定 `PYTHONHASHSEED=0` 后跨 engine 重启 5,872/5,872 对象命中；
  未设置时相同 prompt 重启后 5,748 个候选全部 cold miss。
- warm：每 rank 4.48 GB 用时 1.11–1.18 s，8-rank 有效载荷约
  30.3 GB/s，API 2.72 s。
- server 重启后的 cold disk：每 rank 1.64–1.75 s，双 NVMe 峰值
  2.11 + 2.18 GB/s，API 2.82 s，completion error 增量为 0。
- PR #280 将 5,872-key scheduler `batch_exist` 从 246 ms 降到 71 ms：
  RDMA `ExistMany` 复用 GET fanout 分片，并并行 shared-memory rendezvous
  probe。完整 warm API 为 2.02 s；剩余主耗时在 GPU load/engine barrier。


v2.19 CUDA GET 的正确性 fence 增加了每次操作的临时 pinned host staging、H2D copy、
CUDA stream 同步及相关分配/注册成本；其内存峰值随并发 GET payload 增长。这是安全
direct-GPU retry fencing 落地前的已知性能取舍，不应按旧版 direct-GPU GET 数据外推。

### 3.7 已知问题 / 排查

| 现象 | 原因 / 解 |
|---|---|
| 写成功但**读永不命中** | 未同时设置 `--prefix-caching-hash-algo sha256` 与固定 `PYTHONHASHSEED`（当前连接器会启动失败）；或所有实例 seed 不同；或 effective namespace / canonical object key 不一致 |
| 命中后输出/shape 错误 | 同一 namespace+key 被不同 dtype/page/shape/layout 复用；这是 type-safety violation。停写，bump source-controlled raw-layout ID 并同时发布所有 writer/reader |
| 每个 RDMA `put` 失败 `rc=-1` | `members` 指了 `--port` 而非 `--rdma-port` |
| `ibv_reg_mr` 失败 / 无 GPUDirect | GPU 节点没加载 `nvidia-peermem` |
| CUDA GET pinned allocation / MR 注册失败或吞吐下降 | v2.19 GET 需要按并发 payload 分配临时 pinned host staging，并执行 H2D copy + stream 同步；检查容器/systemd memlock 与宿主机 pinned-memory 容量，压测实际并发和块大小 |
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
生产多 fabric 节点必须用 `DFKV_RDMA_DEV` 配置同一份有序全轨列表。in-process
LMCache 可启用 `rail_affinity=true`，connector 在各 worker native open 前按
`metadata.local_worker_id` 收窄为 primary + fallback；缺少本地 rank 元数据时启动失败。

**🔴 生产多模型 / 多租户隔离（in-process 隔离能力的天花板）**

in-process connector 的 canonical namespace（`remote_connector.py:93-117`）字段中
`model_name` 来自 LMCache upstream `KVCacheMetadata`（必有）；`tenant_id`/`model_revision`
经 `getattr(metadata, "tenant_id", "default")` 读取——**上游 KunCacheMetadata 不含这两个
字段**，因此 in-process **`tenant_id` / `model_revision` 全部固定为 `"default"`，无法配置**。
**layout 参数变化不能用 revision 隔离**。

因此 LMCache in-process **只暴露一个隔离维度**：

| 维度 | 配置位点 | 示例 |
|---|---|---|
| ① Group（ring） | **`lmcache.yaml` 的 `url`**（`dfkv://<mds列表>/<group>`）| `dfkv://10.201.3.10:28150/glm-prod-chat` |
| ③ tenant | ❌ 不可配 | — |
| ④ revision | ❌ 不可配 | — |

**生产两套模型/两业务线和配置变化的 tenant 隔离 → 用 MP-server L2 adapter（§4.5）**
（全部字段可设）；in-process 场景要用，只能按业务线分 **group**（同 ring 硬件共
享但 namespace=业务隔离）：

```yaml
# 业务 chat
remote_storage_plugin.dfkv.url: dfkv://10.201.3.10:28150,10.201.3.11:28150/glm-chat

# 业务 coder (同模型、同布局时)
remote_storage_plugin.dfkv.url: dfkv://10.201.3.10:28150,10.201.3.11:28150/glm-coder
```

🔴 **同 group 同 model_name 参数变化（dtype/量化/层宽度任一变）不允许同 key 复用，
必须分新的 `<新名>` group tag**（如 `glm` → `glm-nvfp4-m2`）——因为
in-process 没有 revision 字段可以在编码路径起隔离效应。换一种理解：
**in-process 的 group tag 包揽 tenant+revision 应该充填的 namespace 角色**。

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
   `Discovered adapter: DfkvConnectorAdapter` / `DfkvConnector ready: membership=... endpoint=... rdma_pools=1 ...`。
   affinity 启用时还必须逐 worker 出现
   `LMCache dfkv rail affinity: ... local_rank=N/... selected=... primary=...`。
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
| `rail_affinity` | 否 | 默认 `false`；仅 in-process 路径按 LMCache `local_worker_id` 选择 per-worker primary |
| `rail_affinity_fallbacks` | 否 | 默认 `1`；相邻有序 fallback 数，`0` 为严格 one-rank/one-rail |

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

`adapter_params` 完整字段（全部可选除非标注必填，经 `from_dict` 校验
`l2_adapter.py:150-165`）：

| 字段 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `url` | ✅ | — | `dfkv://<endpoint>/<group>`。`membership="mds"`（默认）时 `<endpoint>` 是**逗号分隔的 MDS ip:port 列表**（多副本自动 failover/round-robin），`group` 是 client 可见的 ring 名。例：`"dfkv://10.201.3.10:28150,10.201.3.11:28150/glm"`。`membership="static"` 时 `<endpoint>` 是字面 member 字符串（如 `"smoke0064=192.168.5.1:28301"`），`group` 同义 |
| `membership` | — | `"mds"` | `"mds"`（生产推荐，动态发现 server）或 `"static"`（简单/测试场景） |
| `lib` | — | `DFKV_LIB` env / 系统路径 | `libdfkv.so` 路径（必须 v2 SOVERSION，无 v1.x 兼容） |
| `model_name` | ✅ | — | **MP-server 不会从 runtime metadata 自动提供**，须显式写（见下方「多模型隔离」约束），影响 canonical namespace |
| `tenant_id` | — | `"default"` | tenant 配额/billing 身份（`docs/ARCHITECTURE.md` §8），同模型多业务共用环时可显式区分（§5 key 隔离 §①） |
| `model_revision` | — | 同 `model_name` | 模型 revision/版本标签——同 model_name 但权重/布局变化时**必须区分**（§5 key 隔离 §②） |
| `mds_poll_ms` | — | `3000` | 客户端 MDS 环重新发现间隔（毫秒） |
| `num_workers` | — | `8` | 客户端 I/O 并发度 |
| `max_capacity_gb` | — | `0` | > 0 时启用 LMCache 聚合 L2 切割（见 §4.6.6）；0 = 容量交给 dfkv 自管 |

**MDS 列表的填写规则**：全部可用 MDS 副本的 `ip:port` 以逗号拼接，无空格。
client 启动时挨个尝试，失败自动下位。若以后只改了一处且其他已死/未在，
该副本依旧可用（membership 由 lease 管投入，不由列表管断路）。
示例（生产 3 副本 MDS 署）：

```json
"adapter_params":{
  "url":"dfkv://10.201.3.10:28150,10.201.3.11:28150,10.201.3.12:28150/glm",
  "membership":"mds",
  "lib":"/dfkv/lib/libdfkv.so",
  "model_name":"glm-5.2",
  "model_revision":"2026-08-04-nvfp4-m2",
  "tenant_id":"prod-chat",
  "num_workers":16}
```

#### 多模型/多租户配置（对应 §5 四维度）

| 维度 | 配置位点 | 示例 |
|---|---|---|
| ① Group | `url` 末尾 `/<group>` | `dfkv://10.201.3.10:28150/glm-prod-chat` |
| ③ tenant_id | `adapter_params.tenant_id` | `"tenant_id":"prod-chat"` |
| ④ model_name | `adapter_params.model_name`（**必填**） | `"model_name":"glm-5.2"` |
| ④ model_revision | `adapter_params.model_revision` | `"model_revision":"nvfp4-2026-08"` |

完整生产示例（同模型同参数量化，两业务线分 group + 分 tenant）：

```json
// chat
{
  "adapter_params":{
    "url":"dfkv://10.201.3.10:28150,10.201.3.11:28150/glm-chat",
    "model_name":"glm-5.2",
    "model_revision":"nvfp4-2026-08",
    "tenant_id":"prod-chat"}
}
// coder
{
  "adapter_params":{
    "url":"dfkv://10.201.3.10:28150,10.201.3.11:28150/glm-coder",
    "model_name":"glm-5.2",
    "model_revision":"nvfp4-2026-08",
    "tenant_id":"prod-coder"}
}
```

**vs in-process**：只有 MP-server L2 adapter 能配置 tenant_id/model_revision；
in-process 上游 metadata 无这两项（见 §4.3 🔴）。对生产参与 tenant 配额或
参数变化隔离场景的**唯一选项是本 adapter**。MP-server API 不提供足以证明 byte-identical MLA replica 的 model/PP
元数据，因此对象 key 始终保留 `world_size/global_rank` identity，并拒绝手工折叠开关。
server 的 pinned L1 arena 在 LMCache 传入 `l1_memory_desc` 时自动注册 RDMA 零拷贝。

实现要点：dfkv 无原生 eventfd，`DfkvL2Adapter` 用**后台 asyncio loop + 三个
`create_event_notifier`（store/lookup/load）+ done-callback** 把同步 ctypes 客户端桥接到
`L2AdapterInterface`；`ObjectKey` 经统一 codec 渲染为 §1.4 的 canonical object key。
已在 GLM-5.2（vLLM 0.23.0 + LMCache 0.4.7）真机验证：store → 重启（L1 清空）→ 从 dfkv
回载、prefill 跳过。单测 `integration/lmcache/tests/test_l2_adapter.py`（fake client）+
集成测试 `test_l2_adapter_integration.py`（`DFKV_L2_URL`/`DFKV_L2_MEMBERSHIP` 指向真环）。

MP-server L2 adapter 只有一个共享 dfkv client，构造时没有单一物理 GPU local
rank，因此不应用 rank-local affinity；`adapter_params` 中也不接受这两个键。
需要 per-worker/HCA 映射时使用 §4.1–4.4 的 in-process 路径。

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
stored length 不等而 miss。变长 get 解决这个问题（另注：GET 目标 buffer
不可写/非连续时 native 层 fail-loud 抛错，向上冒泡为该 chunk miss，而非假命中，
见 **②** 的 pointers 一条）：
- C++：`KVClient::GetAuto(key, out, cap, *out_len)` /
  `BatchGetAuto(items, *out_lens)`；只要实际长度 `<= cap` 就返回 raw bytes 与长度。
- C ABI：`dfkv_get_auto` / `dfkv_batch_get_auto`。
- `BatchGetAuto` 复用 RDMA 零拷贝 `RangeInto` 路径：buffer 容量是上限，
  payload 直接散射进调用方 buffer，实际长度另行回传；connector 再调
  `reshape_partial_chunk(memory_obj, bytes_read)` 裁剪 shape。

**② pybind11 → ctypes。** dingofs 用 pybind11 原生模块（eventfd 完成队列）；dfkv 直接
ctypes 调 `libdfkv.so`（与 HiCache 插件一致）。C ABI 同步且内部线程安全：`dfkv_batch_*`
阻塞、内部线程池跨 owning node 并行 fan-out；成员 ring 有互斥锁，**一个 `dfkv_open_v2`
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
4. 零拷贝指针：`(c_char*nbytes).from_buffer(mv)` 直接别名可写连续 buffer。PUT 对
   只读 buffer 退回 `from_buffer_copy` staging（数据先拷入 staging，再发 native）；
   **GET 不作 staging**——目标 buffer 不可写或非 C 连续时，在**所有 native 调用之前**
   `raise ValueError`（fail-loud，连接器上层把它当 miss 处理），绝不返回指向
   staging 副本的假命中。keepalive 对象保活到 C 调用返回。
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

### 4.7 实测结果（历史实测，参考）

> ⚠️ 下表是 **vLLM 0.21 + LMCache 0.4.5** 旧栈上的历史实测（相对收益方向仍
> 可参考），v2.0.0 原生身份/传输栈下的绝对值需以新实测与新部署为准。

环境：a100（vLLM 0.21 + LMCache 0.4.5，历史实测）；dfkv = 2 节点 static 成员（TCP 18800 / RDMA
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

- **vLLM 在请求被中止时崩溃**（历史实测栈 LMCache 0.4.5 + vLLM 0.21 观察）：在该旧栈上 `FINISHED_ABORTED` 时
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

### 5.1 多模型/多租户生产隔离四维度

生产环境跑多套模型或同模型多业务时，按四个维度组合隔离（独立起效，全组合生效）。
**建议全部四列都设**，否则撞上跨写/错读开头后很难查。

#### ① Group（环归属）

同一个 etcd 里多个 group = 各自的 membership/ring。**不同业务线分 group**
（`mds_group`；in-process LMCache 的 `url` 尾段；MP-server 的 `url` 尾段）。
server 只入一个 group（flag），client 只看到本组 server，无跨组走 wire 的场景。
特殊场景（如 K3 KV pool 供同一模型多个 DP rank 共享）则有意共用同 group。

#### ② canonical namespace（模型身份 + 布局）

每条 connector 启动时浇出一个 sources-controlled namespace descriptor：
`model_identity` + `model_revision` + `tenant_id` + `dtype` + `block_tokens` +
`num_layers` + `tp/dp/pp_size` + `group_layout`。这些都进 BlockKey 的 SHA-256，任一不同
均冷 miss。**同名模型但参数/布局不同（dtype、量化、TP 组布局、header/cache dtype）必须
靠显式 `model_revision` 或 `tenant_id` 隔离**，否则高概率复用到不可解释的
bytes（见下）。

#### ③ tenant_id（配额与计费）

`tenant_id` 进 BlockKey（64-bit tenant hash）**且**进配额 admission（default+strict hash
表，`dfkv_tenant_quota.py` 管理）。生产多模型共环时，为每个业务线分配一个
tenant_id（如 `prod-chat`、`prod-coder`、`dev-ab`），可以同时拿到
*该业务的 usage 配额*与*身份隔离*。tenant_id 只影响 identity/quota，**不构成 ACL**——
任何能到达端口的 client 都可以自报 tenant_hash（`docs/ARCHITECTURE.md` §8）。

#### ④ model_name / model_revision（显式模型版本标签）

- `model_name`：精确模型 identity（如 `glm-5.2`、`deepseek-v2-lite`）。MP-server/
  in-process LMCache 必填（不自动从 runtime metadata 提供）。
- `model_revision`：同 `model_name` 但权重、量化、并发点切边时间不同（如
  `2026-08-04-nvfp4-m2`、`fp8-ep16-团子`）时的**必修标签**。默认值 = `model_name`；
  不同时**必须显式设置**，否则同 model_name 不同布局会命中成 byte-incompatible 冷数据。

### 5.2 三套生产配置实例

**A. 同模型不同业务线（同 group + 不同 tenant）**

```jsonc
// 业务 chat: url="dfkv://mds.../glm", tenant_id="prod-chat",
//                model_name="glm-5.2", model_revision="nvfp4-2026-08"
// 业务 coder: url="dfkv://mds.../glm", tenant_id="prod-coder",
//                 model_name="glm-5.2", model_revision="nvfp4-2026-08"
```
**效果**：同一 ring；quota 独立；key identity 互不可见（tenant hash 不同）。
生产建议：为每个 tenant_id 在 `dfkv_tenant_quota.py` 预置配额。

**B. 同模型不同参数量化（同 group + 不同 model_revision）**

```jsonc
// v1: model_name="deepseek-v4-flash", model_revision="fp8-ep8-0731"
// v2: model_name="deepseek-v4-flash", model_revision="nvfp4-0731"
```
**效果**：同 ring；identity 互 miss；不会被拉错 layout。布局切换（如 FP8
屁股换 NVFP4）必须 bump model_revision，不允许沿用默认。

**C. 业务线与模型交叉（多 group + 多 tenant）**

```jsonc
// chat:  mds_group=glm-chat,  tenant_id=prod-chat,   model_name="glm-5.2"
// coder: mds_group=glm-coder, tenant_id=prod-coder,  model_name="glm-5.2"
// k3:    mds_group=gcp-chat,  tenant_id=kimi-k3,     model_name="kimi-k3"
```
**效果**：三 ring；各 server 只入自身 group；互不可见（group 本身就是权限边）。
注意 server 只入一个 group——多业务线要么加节点独属 group，要么共用 group 靠 A/B 两
种组合区分。

### 5.3 排障看 namespace 的第一眼

冷缓存/miss 但 server 写入看起来正常的首要排查对象就是 namespace：

1. **init 行**（access log，见 access_log.md §1）：核对 `init(r0 <model> ...) : ok <membership>` 里的 model 与 endpoint 是否对得上该实例应入的 group/预期 revision。
2. **client INFO**（`dfkvctl clients --group <g>`）：现册 client 自报的 model_hash（tenant hash 的入口）与 target model_name/model_revision 是否匹配。
3. **命中率的 identity 欺诈**（hit_rate_funnel.md §①）：model_identity/pcp/dcp/
  group_layout 任一与预期不同但在同一 MDS group，可用 `dfkv_mds_group_version_skew`
  指标看环内 client 是否跑错版本。

### 5.4 🔴 对生产环境的硬性警告

1. **布局变化必须 bump 显式 revision**：dtype、量化、层数、TP/PCP/DCP/PP 维度任一
变化都改变 layout。`model_revision`（connector 级）或 connector 内部
source-controlled layout ID 必须同步更新；dfkv 本身不保存 header。
2. **v2 与 v1.x 不能混 wire**：v2 client/server 只收 epoch 6/7 frame，v1.40 前的
frame 会被 fast-reject（`wire.h:92-108`）。混合代际升级走 MDS 双协议
（DFKV_MDS_ACCEPT_LEGACY）的受控迁移，不允许双写数据面。
3. **operator 自定义 namespace 别名被拒**（03d7035）：v2.0.0 起 connector 只接受
source-controlled layout ID，不允许 runtime 传入部署标签当 payload schema。
4. **🔴 跨实例 L3 复用必须全实例 `PYTHONHASHSEED=0`**：vLLM block hash 跨进程
确定性靠固定种子；不设时跨实例命中率呈静默 0（与 replicated-MLA 修复前同签名）。

---

## 相关文档

- [DEPLOY.md](DEPLOY.md) — dfkv 集群（server + MDS）标准部署
- [ARCHITECTURE.md](ARCHITECTURE.md) — 存储/协议架构（wire 协议、slab、RAM 热层）
- [datapath-perf-notes.md](datapath-perf-notes.md) — RDMA 数据面性能笔记（depth-flat、SG coalescing）
- [access_log.md](access_log.md) / [METRICS.md](METRICS.md) / [tracing.md](tracing.md) — 可观测性
- `integration/vllm/README.md` · `integration/lmcache/README.md` — 包内快速参考
- `docs/superpowers/specs/2026-06-18-dfkv-vllm-store-connector-design.md` — vLLM 连接器设计文档
