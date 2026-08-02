# dfkv 连接器分布式追踪（Distributed Tracing）

> 三个连接器（SGLang HiCache / vLLM 直连 / LMCache）**同一套配置、同一套 span 模型**。
> 默认**关闭**，开启后对热路径近乎零开销；只有被采样的请求才会真正生成并上报 span。

给每个连接器的**数据面 op**（`batch_get_v1`/`batch_set_v1`/`batch_exists`/`batch_get`/
`batch_set`/`batch_exist`/`batch_remove` …）按需生成一条 [OTLP](https://opentelemetry.io/)
span，经 `OTEL_EXPORTER_OTLP_ENDPOINT` 的 `/v1/traces` 推到中心 Collector，再进
Jaeger / Tempo 等追踪后端。每条 span 带连接器身份（connector_id / type / host / tp_rank /
version）作为 OTLP **resource** 属性，所以多实例 / 多 TP rank / 多机的 span 会汇聚到同一个后端
——这就是「分布式」视角。

> **范围**：当前为**连接器侧 span**（每个 op 一条根 span）。wire 协议与 C++ cache server
> 未改动；把 trace 上下文跨进程传播到 server、由 server 生成子 span，是后续里程碑。

## 上报哪些请求（在 op 结束、已知耗时后判定）

一条请求满足以下**任一**条件即上报：

1. **慢请求** —— 耗时 ≥ `trace_slow_request_ms`（核心：定位长尾 / 卡顿）。
2. **百分比采样** —— 对慢请求之外，再随机采集 `trace_sample_percent`%（0–100）的请求，看常态分布。
3. **失败** —— op 抛异常或被标记为 fail 时**总是**上报（便宜，且正是排障要看的）。

慢请求与百分比是**叠加**的：例如 `slow=1000, percent=1` = 上报所有 ≥1s 的请求 **加** 额外 1% 的常态请求。

## 配置项

**优先级：`extra_config` 键 > 环境变量 > 默认值**（每个开关独立解析，与 access log / 车队指标同口径）。

| 作用 | `extra_config` 键 | 环境变量 | 默认 |
|---|---|---|---|
| 总开关 | `tracing` | `DFKV_TRACING_ENABLED`（或伞开关 `DFKV_TELEMETRY_ENABLED`）| 关 |
| 慢请求阈值（毫秒） | `trace_slow_request_ms` | `DFKV_TRACE_SLOW_REQUEST_MS` | `1000`（`0` = 关闭慢触发）|
| 采样百分比（0–100） | `trace_sample_percent` | `DFKV_TRACE_SAMPLE_PERCENT` | `0`（不额外采样）|
| 推送间隔（毫秒） | `trace_export_interval_ms` | `DFKV_TRACE_EXPORT_INTERVAL_MS` | `5000` |
| span 缓冲上限（条） | `trace_max_buffered_spans` | `DFKV_TRACE_MAX_BUFFERED_SPANS` | `2048`（满则丢最旧并计数）|
| OTLP 端点 | `otlp_endpoint` | `OTEL_EXPORTER_OTLP_ENDPOINT` | 复用车队指标端点，自动追加 `/v1/traces` |

- 开关取值接受 `1/true/yes/on`（大小写不敏感）或 JSON 里的 `1`/布尔。
- 伞开关 `DFKV_TELEMETRY_ENABLED=1` 会**同时**打开车队指标与追踪；想单独控制就用 `DFKV_TRACING_ENABLED`。
- `trace_slow_request_ms=0` 关闭「按耗时上报」，只保留百分比 + 失败上报。
- `trace_sample_percent` 越界自动夹到 `0..100`。
- 端点解析复用车队指标那套：`http://host:4318` → `http://host:4318/v1/traces`；已带 `/v1/traces` 则原样用。
- 连接器侧 vLLM / LMCache **只认环境变量**（不读各自的 extra_config）；SGLang HiCache 还能从启动 JSON 的 `extra_config` 读上表的键。

## span 内容

- span 名 = op 名（`batch_get_v1` 等），`kind=CLIENT`。
- 属性：`op`、`dfkv.keys`、`dfkv.hits`（写成功 / 命中 / present 的数量）、`dfkv.bytes`、
  `dfkv.duration_ms`、`status`（`ok`/`fail`）；失败时再带 `dfkv.error`（异常类型+信息），
  并把 span 的 OTLP `status.code` 置为 ERROR。
- 身份（connector_id / connector_type / host / pid / tp_rank / version / native_version）在
  OTLP **resource** 上，Collector 会转成后端可查的标签，与车队指标共用一套身份。

## 使用方式

### 方式 A — SGLang HiCache 启动 JSON（贴合 dfkv 现有 `extra_config` 约定）

在 `--hicache-storage-backend-extra-config` JSON 里加追踪键（与 access_log 键并列）：

```bash
sglang serve ... \
  --hicache-storage-backend dynamic \
  --hicache-storage-backend-extra-config '{
    "backend_name":"dfkv","module_path":"dfkv_hicache","class_name":"DfkvHiCache",
    "interface_v1":1,
    "mds_endpoints":"10.0.0.1:9400,10.0.0.2:9400","mds_group":"default",
    "otlp_endpoint":"http://collector:4318",
    "tracing":1,
    "trace_slow_request_ms":1000,
    "trace_sample_percent":1 }'
```

SGLang supplies `model_name` from the runtime and combines it with the
source-controlled `sglang-hicache/raw-v1` namespace.

### 方式 B — 环境变量（三连接器通用；vLLM / LMCache 只支持这种）

```bash
DFKV_TRACING_ENABLED=1 \
DFKV_TRACE_SLOW_REQUEST_MS=1000 \
DFKV_TRACE_SAMPLE_PERCENT=1 \
OTEL_EXPORTER_OTLP_ENDPOINT=http://collector:4318 \
<vllm serve ... | lmcache ... | sglang serve ...>
```

> 端点指向你的 OTel Collector；Collector 把 traces 转发给 Jaeger / Tempo。
> 与车队指标共用 `OTEL_EXPORTER_OTLP_ENDPOINT`（metrics 走 `/v1/metrics`，traces 走 `/v1/traces`）。

## 哪些 op 会被追踪

| 连接器 | 被追踪的 op |
|---|---|
| SGLang HiCache | `batch_set_v1`、`batch_get_v1`、`batch_exists`、`batch_set_v2`、`batch_get_v2` |
| vLLM 直连 | `batch_put`、`batch_get`、`batch_put_sg`、`batch_get_auto_sg`、`batch_exist` |
| LMCache | `batch_set`、`batch_get`、`batch_exists`、`batch_remove` |

> 只追踪生产批量接口；generic 单页 `set`/`get` 回退路径与 `batch_exists_v2` 不单独追踪
> （它们内部会调用上表中已追踪的 op，避免产生重复 / 嵌套的根 span）。

## 性能

- **关闭时**：约几十纳秒/次。`span()` 返回冻结的 noop 单例，跳过计时、判定、span 组装与 OTLP 导入。
- **开启但未命中采样**：仅一次 `perf_counter` enter/exit + 一次判定（约百纳秒），**不**生成 trace/span id、不组 span、不分配。
- **命中采样**：才生成 16B trace id / 8B span id + span dict，入有界缓冲；POST 在后台 daemon 线程批量做，**不在热路径**。缓冲满则丢最旧并计数，绝不阻塞请求。进程退出时 `atexit` 自动 flush 一次。
- 与 access log / 车队指标一样，是三条互相独立的旁路开关——可单独开追踪而不开另两者。

## 验证 / 复跑

无需 GPU/torch，纯单测覆盖采样判定、OTLP payload 组装与缓冲丢弃：

```bash
cd test/python && python3 -m unittest test_dfkv_telemetry test_telemetry_vendor_sync -v
```

端到端（起一个本地 Collector 或假 HTTP 端点，设 `DFKV_TRACING_ENABLED=1
DFKV_TRACE_SLOW_REQUEST_MS=0 DFKV_TRACE_SAMPLE_PERCENT=100
OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318`，跑连接器 smoke，确认 `/v1/traces` 收到 span）。

## 实现位置

- [integration/hicache/dfkv_telemetry/tracing.py](../integration/hicache/dfkv_telemetry/tracing.py) — `configure()` / `span()` 上下文管理器、noop 单例、采样判定、span 组装。
- [integration/hicache/dfkv_telemetry/otlp_traces.py](../integration/hicache/dfkv_telemetry/otlp_traces.py) — 纯 stdlib OTLP/HTTP-JSON traces 导出器（有界缓冲 + 后台批量 POST）。
- [integration/hicache/dfkv_telemetry/config.py](../integration/hicache/dfkv_telemetry/config.py) — 追踪开关 / 阈值的 env+extra_config 解析（`tracing_enabled`）。
- 连接器接线：[integration/hicache/dfkv_hicache.py](../integration/hicache/dfkv_hicache.py)、[integration/vllm/src/dfkv_vllm/dfkv_client.py](../integration/vllm/src/dfkv_vllm/dfkv_client.py)、[integration/lmcache/src/dfkv_connector/native_client.py](../integration/lmcache/src/dfkv_connector/native_client.py)。
- 单测：[test/python/test_dfkv_telemetry.py](../test/python/test_dfkv_telemetry.py)（`Tracing*` / `OtlpTraces*` 用例）。

> vLLM / LMCache 连接器各自 vendor 一份逐字节一致的 `_telemetry` 副本（含 `tracing.py`/`otlp_traces.py`），
> 由 [deploy/sync_telemetry.sh](../deploy/sync_telemetry.sh) 同步、`test_telemetry_vendor_sync` 守护防漂移。
