# dfkv 连接器上报使用文档

三个连接器(vLLM / LMCache / SGLang HiCache)如何把 KV cache 的运行指标
**上报(PUSH)** 到中心 observability 栈,在 Grafana 上看。

> 一句话:三个连接器**默认都不上报**(零开销)。打开方式是 `DFKV_METRICS_ENABLED=1`
> + 指向一个 OTLP Collector;差别只在「配置写在哪」—— vLLM / LMCache 用环境变量,
> SGLang HiCache 还能写在 `extra_config` 里。

---

## 0. 数据怎么流的

```
连接器 (vllm/lmcache/hicache) --OTLP push--> otel-collector --+--> prometheus --> grafana
dfkv_server / dfkv_mds  <--Prometheus pull (/metrics)---------/        ^
                                              otel-collector --traces--> tempo --/
```

- **连接器是 PUSH**:它们数量多、动态调度,主动把聚合后的指标用 OTLP 推给 Collector。
- **C++ 守护进程是 PULL**:`dfkv_server` / `dfkv_mds` 暴露 `/metrics`,被 Prometheus 抓。
  本文档只讲连接器(PUSH);C++ 侧见 `README.md` 的 "backend" 面板。

| 连接器 | pip 包 | `connector_type` | 接入引擎方式 | telemetry 配置渠道 |
|---|---|---|---|---|
| vLLM | `dfkv-vllm` | `vllm` | `--kv-transfer-config` | **仅环境变量** |
| LMCache | `dfkv-connector` | `lmcache` | LMCache yaml `remote_storage_plugins` | **仅环境变量** |
| SGLang HiCache | dfkv 仓库内 `integration/hicache/dfkv_hicache.py` | `hicache` | `--hicache-storage-backend dynamic` | **extra_config 或环境变量**(extra_config 优先) |

---

## 1. 通用前提(三个连接器都一样)

### 1.1 得有接收端
连接器只负责推,要有人收。二选一:

- **本地起一套**(自测用):
  ```bash
  export DFKV_GRAFANA_ADMIN_PASSWORD='replace-me'
  docker compose -f deploy/observability/docker-compose.yml up -d
  # podman 无 compose 插件时,容器已存在可直接:
  # docker start otel-collector prometheus grafana tempo
  ```
  默认 stdlib exporter 用 HTTP `localhost:4318`；`localhost:4317` 只给显式
  OTel SDK gRPC exporter。Grafana 为 `localhost:3000`。
- **指向中心化 Collector**：把下面的 `OTEL_EXPORTER_OTLP_ENDPOINT` 换成中心 HTTP receiver 地址即可。

### 1.2 依赖(默认零依赖,不用装任何东西)
telemetry 是 opt-in,且**默认用纯 stdlib 的 OTLP/HTTP-JSON 推送器**(`DFKV_METRICS_EXPORTER=stdlib`,默认)——
**开 `DFKV_METRICS_ENABLED=1` 就能推,不需要装任何第三方包**。

只有当你想改用 OpenTelemetry SDK 导出器(`DFKV_METRICS_EXPORTER=otel`)时,才装可选依赖:

```bash
pip install 'dfkv-vllm[otel]'          # vLLM
pip install 'dfkv-connector[otel]'     # LMCache
# SGLang HiCache 随 dfkv 仓库走,单独装 OTel SDK:
pip install opentelemetry-sdk opentelemetry-exporter-otlp
```
> 不装也能推(走 stdlib);装了且设 `DFKV_METRICS_EXPORTER=otel` 才用 SDK。metrics 关时两者都是零成本 no-op。

### 1.3 通用配置项
全部 opt-in。环境变量名 + (SGLang 才支持的)`extra_config` 键 + 默认值:

| 环境变量 | extra_config 键 | 默认 | 作用 |
|---|---|---|---|
| `DFKV_METRICS_ENABLED` | `metrics` | `0`(关) | 推送指标总开关 |
| `DFKV_TELEMETRY_ENABLED` | `telemetry` | `0` | 大开关（metrics + traces）；各自显式开关优先 |
| `DFKV_METRICS_EXPORTER` | `metrics_exporter` | `stdlib` | 导出器:`stdlib`(纯标准库 OTLP/HTTP-JSON,**零依赖,默认**) 或 `otel`(用 OpenTelemetry SDK,需装 `[otel]`) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `otlp_endpoint` | stdlib: `http://localhost:4318`；otel: SDK 默认 | Collector endpoint；默认 stdlib 只发 OTLP/HTTP JSON（通常 `:4318`），显式 SDK exporter 才可按 SDK 配置 gRPC `:4317` |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `otlp_protocol` | stdlib: 固定 HTTP/JSON；otel: SDK 默认 | 只对 `DFKV_METRICS_EXPORTER=otel` 生效；stdlib 不读取它 |
| `DFKV_CONNECTOR_ID` | `connector_id` | `<host>_<pid>_<tp_rank>` | 实例标识（建议给稳定可读值；字符限 `[A-Za-z0-9._-]`） |
| `DFKV_MODEL` | `model` | runtime model / empty | Fallback model label when the connector does not pass authoritative runtime identity |
| `DFKV_DEPLOYMENT` | `deployment` | empty | Stable deployment/serving-pool label |
| `DFKV_CACHE_ROLE` | `cache_role` | runtime role / empty | `kv_producer` / `kv_consumer` / `kv_both` or the deployment's equivalent |
| `DFKV_TEAM` | `team` | empty | Owning team label |
| `DFKV_METRICS_EXPORT_INTERVAL_MS` | `metrics_export_interval_ms` | `10000` | OTLP 推送间隔(最小 1000) |
| `DFKV_PROBE_INTERVAL_MS` | `probe_interval_ms` | 关(开 metrics 时 `5000`) | C++ 主动逐 peer 延迟探测,空闲节点也出延迟 |
| `DFKV_PEER_LATENCY_POLL_S` | `peer_latency_poll_s` | `10` | 逐 peer 延迟 snapshot→push 间隔 |

> **成本与生命周期**：metrics 关时，连接器 op 路径不求值任何指标参数
>（falsy 短路），不 import OTel、不启动线程。打开后，每个 op 只更新内存聚合，
> 后台按 `DFKV_METRICS_EXPORT_INTERVAL_MS` 推送。多个同进程 connector 对共享
> recorder 做引用计数；最后一个 close 会停止 poller、串行执行一次 final push，
> 再释放 exporter。周期 push 失败不会丢掉窗口最大值或 native counter delta，
> 下一次成功会重放。

---

## 2. vLLM 连接器(`dfkv-vllm`)

### 2.1 接入引擎
vLLM 通过 `--kv-transfer-config` 加载,`kv_connector_extra_config` 配 dfkv 连接参数:

```bash
vllm serve <model> \
  --kv-transfer-config '{"kv_connector":"DfkvStoreConnector",
    "kv_connector_module_path":"dfkv_vllm.connector",
    "kv_role":"kv_both",
    "kv_connector_extra_config":{"members":"c1=<ip>:<rdma-port>",
      "lib":"/path/to/libdfkv.so"}}'
```

vLLM supplies the exact runtime `model_name` and combines it with the
source-controlled `vllm/raw-v1` namespace.

### 2.2 打开 telemetry
**只认环境变量**(连接器内部 `configure({})`,**不会**去读 `kv_connector_extra_config`),
所以把开关放进 vLLM 进程的环境里:

```bash
export DFKV_METRICS_ENABLED=1
export OTEL_EXPORTER_OTLP_ENDPOINT=http://<collector>:4318
export DFKV_CONNECTOR_ID=prefill-rank0          # 可选,建议给
# 然后照常 vllm serve ...
```

> 注意:**别**想把 telemetry 配进 `kv_connector_extra_config`,那里对 telemetry 无效。
> 想要空闲节点也出逐 peer 延迟,需自己 `export DFKV_PROBE_INTERVAL_MS=5000`(vLLM 不自动开)。

---

## 3. LMCache 连接器(`dfkv-connector`)

### 3.1 接入引擎
LMCache 配置(plugin 模式,推荐):

```yaml
remote_storage_plugins: ["dfkv"]
extra_config:
  remote_storage_plugin.dfkv.module_path: dfkv_connector.adapter
  remote_storage_plugin.dfkv.class_name:  DfkvConnectorAdapter
  remote_storage_plugin.dfkv.url:         dfkv://mds1:6700,mds2:6700/group-1
  # Other optional control metadata: membership(mds|static)、lib、mds_poll_ms
```

### 3.2 打开 telemetry
同样**只认环境变量**(连接器内部 `configure({})`,不读 LMCache 的 `extra_config`)。
在跑 LMCache 的进程环境里:

```bash
export DFKV_METRICS_ENABLED=1
export OTEL_EXPORTER_OTLP_ENDPOINT=http://<collector>:4318
export DFKV_CONNECTOR_ID=lmcache-node1          # 可选
```

> LMCache 之前没有 dfkv 指标,这一层让它和 vLLM/SGLang 拉平。
> 同样,逐 peer 延迟探测需手动 `export DFKV_PROBE_INTERVAL_MS=5000`。

---

## 4. SGLang HiCache 连接器(`integration/hicache/dfkv_hicache.py`)

### 4.1 接入引擎
SGLang 通过 `--hicache-storage-backend dynamic` 加载,**所有配置走 `extra_config`**
(SGLang 的 hicache storage 配置块),最少要有 `interface_v1=1` + 集群发现(`mds_endpoints` 或 `members`):

```jsonc
{
  "interface_v1": 1,                       // 必须,选 zero-copy v1 RDMA 路径
  "mds_endpoints": "mds1:6700,mds2:6700",  // 或 "members": "c1=<ip>:<port>"
  "lib_path": "/path/to/libdfkv.so"
  // ... 其它 dfkv 连接参数
}
```

SGLang supplies the exact runtime `model_name` and combines it with the
source-controlled `sglang-hicache/raw-v1` namespace.

### 4.2 打开 telemetry —— 两种写法,二选一
HiCache 连接器把 `extra_config` 传进了 telemetry(`configure(cfg)`),所以
**extra_config 键和环境变量都行,extra_config 优先**。

**写法 A:写进 extra_config(和连接参数放一起,推荐)**
```jsonc
{
  "interface_v1": 1,
  "mds_endpoints": "mds1:6700,mds2:6700",
  "metrics": 1,                              // ← 打开上报
  "otlp_endpoint": "http://<collector>:4318",
  "connector_id": "sglang-tp0",              // 可选
  "probe_interval_ms": 5000                  // 可选,见下
}
```

**写法 B:用环境变量**(和 vLLM/LMCache 一样,extra_config 没写时生效)
```bash
export DFKV_METRICS_ENABLED=1
export OTEL_EXPORTER_OTLP_ENDPOINT=http://<collector>:4318
```

### 4.3 SGLang 特有:自动开逐 peer 延迟探测
只要 metrics 打开,HiCache 连接器会**自动**启用 C++ 客户端的逐 peer 延迟探测
(默认 `5000ms`),这样即使某些 cache 节点空闲,Grafana 上也能看到它的 avg/max 延迟。
想改用 `probe_interval_ms` / `DFKV_PROBE_INTERVAL_MS`,设 `0` 关闭。
(vLLM / LMCache 不会自动开这个。)

---

## 5. 验证上报通没通

1. **Collector 收到了吗** —— Collector 把收到的指标重新暴露在 `:8889`:
   ```bash
   curl -s http://<collector>:8889/metrics | grep dfkv_connector
   ```
   能看到 `dfkv_connector_info{...} 1` 就说明推上来了。
2. **Grafana** —— 打开 "**dfkv — cluster overview**" 面板,约 15s 内出现实例;
   顶部用 **Connector type / Connector id** 模板变量下钻到某个实例。
   inventory 表里能看到每个实例的 `dfkv_version`(连接器包版本)和
   `dfkv_native_version`(libdfkv.so 版本),滚动升级时的版本错位会一目了然。

连接器上报的指标:

| 指标 | 类型 | 标签 |
|---|---|---|
| `dfkv_connector_info` | gauge(=1) | 身份:`dfkv_connector_id/type/host/pid/tp_rank` + 版本:`dfkv_version`、`dfkv_native_version` |
| `dfkv_connector_ops_total` | counter | `op`、`status` |
| `dfkv_connector_keys_total` | counter | `op` |
| `dfkv_connector_bytes_total` | counter | `op` |
| `dfkv_connector_op_seconds` | histogram | `op` |
| `dfkv_connector_op_max_seconds` | gauge | `op` |

---

## 6. 关掉上报
不设 `DFKV_METRICS_ENABLED` / `metrics`(或显式设 `0`)即可,连接器回到零开销 no-op,
不装 `[otel]` 依赖也不影响连接器正常工作。

## 速查

> 默认 stdlib 导出器**零依赖**——下面"最小开启"都不用 pip install;只有想用 OTel SDK 才装 `[otel]` 并设 `DFKV_METRICS_EXPORTER=otel`。

| 我在用 | 最小开启(默认 stdlib,零依赖) |
|---|---|
| vLLM | `export DFKV_METRICS_ENABLED=1 OTEL_EXPORTER_OTLP_ENDPOINT=http://<collector>:4318`（想用 SDK 才 `pip install 'dfkv-vllm[otel]'`） |
| LMCache | 同上两个 env（想用 SDK 才 `pip install 'dfkv-connector[otel]'`） |
| SGLang HiCache | extra_config 加 `"metrics":1,"otlp_endpoint":"http://<collector>:4318"`（想用 SDK 才 `pip install opentelemetry-sdk opentelemetry-exporter-otlp`） |
