# dfkv 连接器 access log（访问日志）

> 本文以 SGLang HiCache（`DfkvHiCache`）为例；**vLLM 直连连接器与 LMCache 连接器是同一套环境变量（`DFKV_ACCESS_LOG_*`）、同一行格式**，对应配置见 [CONNECTORS.md](CONNECTORS.md) §1.6。

给 `DfkvHiCache` 插件的**每个继承接口**逐次操作打一行日志，用于排查命中率、慢操作、失败。
默认**关闭**，开启后近乎零热路径开销（异步落盘）。格式对齐 dingofs lmcache / vfs 的 access log：

```
<时间>.<毫秒> <op>(<参数>) : <结果> <耗时秒>
```

实测样例（一个 TP rank 的文件，含 `init` 与各数据面接口）：

```
2026-06-15 20:16:16.727 init(r0 glm-5.1 tp=0/8 mla=1) : ok static <0.000461>
2026-06-15 20:16:16.734 batch_set_v1(r0 3 keys) : ok 3/3 <0.005716>
2026-06-15 20:16:16.734 batch_get_v1(r0 3 keys) : hits=3/3 <0.000392>
2026-06-15 20:16:16.734 batch_exists(r0 3 keys) : prefix=2/3 <0.000110>
2026-06-15 20:16:16.735 set(r0 g0, 4.00KiB) : ok <0.000271>
2026-06-15 20:16:16.735 get(r0 g0, 4.00KiB) : hit <0.000153>
2026-06-15 20:16:16.735 get(r0 g_missing, 4.00KiB) : miss <0.000026>
2026-06-15 20:16:16.735 exists(r0 g0) : found <0.000036>
2026-06-15 20:16:16.735 exists(r0 g_missing) : not_found <0.000034>
```

每行 `<参数>` 都以 `r{tp_rank}` 开头，合并多 rank 文件查看时也能区分来源。

## 配置项

**优先级：`extra_config` 键 > 环境变量 > 默认值**（按每个开关独立解析）。

| 作用 | `extra_config` 键 | 环境变量 | 默认 |
|---|---|---|---|
| 开 / 关 | `access_log` | `DFKV_ACCESS_LOG_ENABLED` | 关 |
| 日志路径 | `access_log_path` | `DFKV_ACCESS_LOG_PATH` | 空 → 写 stderr |
| 慢操作阈值（微秒） | `access_log_threshold_us` | `DFKV_ACCESS_LOG_THRESHOLD_US` | `0`（全记） |
| 单文件滚动大小（字节） | `access_log_max_bytes` | `DFKV_ACCESS_LOG_MAX_BYTES` | `134217728`（128MiB） |
| 保留的滚动备份份数 | `access_log_backup_count` | `DFKV_ACCESS_LOG_BACKUP_COUNT` | `5` |

- 开关取值接受 `1/true/yes/on`（大小写不敏感）或 JSON 里的 `1`/布尔。
- `access_log_threshold_us` 设成比如 `1000`，则只记录耗时 ≥ 1ms 的操作，排查长尾很方便。
- 配置在插件 `__init__` 里**解析一次**作为「启动基线」（每进程首个实例生效，幂等）；不像 dingofs 在 import 时读 env。
- **不重启也能开关**：设了 `DFKV_HOT_CONFIG` 后，`access_log` / `access_log_path` / `access_log_threshold_us` 等可通过控制文件**运行时热生效**，见下文 [运行时热开关](#运行时热开关不重启-sglang)。

## 日志滚动与清理（自动，无需 logrotate）

写文件时走 `RotatingFileHandler`，**按大小自动滚动并清理旧日志**，开着不管也不会撑爆盘：

- 单文件写到 `access_log_max_bytes` 就滚动：`access.log` → `access.log.1` → `access.log.2` …，
  超过 `access_log_backup_count` 份后**覆盖最旧的**。
- 每个 TP rank 的磁盘占用上限 = `max_bytes × (backup_count + 1)`。默认 `128MiB × 6 ≈ 768MiB/rank`，到顶即循环覆盖。
- 每个 rank 各写各的文件、各滚各的，**无跨进程滚动竞争**。
- **逃生口**：`access_log_max_bytes=0` 关闭滚动，退化成单个无界文件（旧行为，需自己接 logrotate 清理）。
- 滚动落盘动作在后台 `QueueListener` 线程里做，**不在热路径**。

## 使用方式

### 方式 A — SGLang 启动 JSON（推荐，贴合 dfkv 现有 `extra_config` 约定）

在 [DEPLOY.md](DEPLOY.md) 的 `--hicache-storage-backend-extra-config` JSON 里加 access_log 三个键：

```bash
sglang serve ... \
  --hicache-storage-backend dynamic \
  --hicache-storage-backend-extra-config '{
    "backend_name":"dfkv","module_path":"dfkv_hicache","class_name":"DfkvHiCache",
    "interface_v1":1,
    "mds_endpoints":"10.0.0.1:9400,10.0.0.2:9400","mds_group":"default",
    "key_namespace":"<optional-coordinated-schema-override>",
    "access_log":1,
    "access_log_path":"/var/log/dfkv/access.log",
    "access_log_threshold_us":0,
    "access_log_max_bytes":134217728, "access_log_backup_count":5 }'
```

SGLang supplies `model_name` from the runtime. Omit `key_namespace` to use the
safe automatic model + `sglang-hicache/raw-v1` namespace.

### 方式 B — 环境变量（不想改启动 JSON 时，运维临时开启）

```bash
DFKV_ACCESS_LOG_ENABLED=1 \
DFKV_ACCESS_LOG_PATH=/var/log/dfkv/access.log \
DFKV_ACCESS_LOG_THRESHOLD_US=0 \
DFKV_ACCESS_LOG_MAX_BYTES=134217728 \
DFKV_ACCESS_LOG_BACKUP_COUNT=5 \
sglang serve ...
```

> 路径目录需提前存在且可写；打开失败会打印一行告警并**回退到 stderr**，不影响服务。

## 路径与多 TP rank

SGLang 每个 TP rank 一个进程，会**同时写同一路径**。为避免互相覆盖/交错，路径按以下规则处理：

- 路径含 `{rank}` / `{model}` 占位符 → 直接替换。
  例：`/var/log/dfkv/access.{rank}.log` → `access.0.log`、`access.1.log` …
- 路径**不含**占位符 → 自动追加 `.r{rank}` 后缀。
  例：`access.log` → `access.log.r0`、`access.log.r1` …

所以上面方式 A/B 配 `access.log`，rank 0 的实际文件是 `access.log.r0`。

## 运行时热开关（不重启 SGLang）

access log 的启动基线在插件 `__init__` 解析后，`access_log()` 每次调用**都读模块级开关**，所以只要能在运行时翻转开关就能免重启生效。dfkv 用**控制文件 + 后台 watcher 线程**实现（`dfkv_hot_config.py`），不占端口、不用信号、TP8 八个 rank 读同一文件一次全生效。

**默认关闭（opt-in）**：只有显式设置了控制文件路径（`DFKV_HOT_CONFIG` 环境变量或 `extra_config` 的 `hot_config_path`）才会启动 watcher；没设则一切照旧、零新线程零开销。

> ✅ **三个连接器都支持**（SGLang HiCache / vLLM 直连 / LMCache）——同一个 `DFKV_HOT_CONFIG` 控制文件、同一套语义。vLLM/LMCache 连接器在客户端 `__init__` 里读启动基线（env 驱动）后即挂 watcher；控制文件里同样只有 `access_log` 家族这些观测 knob 会热生效。

| 作用 | `extra_config` 键 | 环境变量 | 默认 |
|---|---|---|---|
| 控制文件路径（**设了才开热加载**） | `hot_config_path` | `DFKV_HOT_CONFIG` | 空 → 热加载关闭 |
| 轮询周期（秒） | `hot_config_poll_s` | `DFKV_HOT_CONFIG_POLL_S` | `5.0`；`≤0` 关闭 |

**用法**：启动时只需多给一个控制文件路径（值本身可以先不存在）：

```bash
DFKV_HOT_CONFIG=/etc/dfkv/hot.json \
DFKV_ACCESS_LOG_PATH=/var/log/dfkv/access.log \
sglang serve ...        # 启动时 access_log 仍是关的
```

事故来了要抓日志时，**不重启**，写一下控制文件即可（最多 `hot_config_poll_s` 秒后八个 rank 全部生效）：

```bash
# 开（只抓 ≥1ms 的慢调用，减量）
echo '{"access_log": true, "access_log_threshold_us": 1000}' > /etc/dfkv/hot.json
# 排查完关掉：删文件即完全回到启动基线（此处基线是「关」）
rm -f /etc/dfkv/hot.json
```

**语义**：控制文件是「实时覆盖层」，叠加在启动基线之上——

- 文件里**出现**的键覆盖启动值；**缺失**的键回落到启动基线值。
- **删除文件 = 清空覆盖**，所有热 knob 干净回到启动基线（上例即回到「关」）。
- 文件格式错误会被**忽略**（保留上一份好配置）并下个周期重试，不会中断服务。
- `access_log_path` / 滚动参数变了会安全重建 sink（旧 listener 先停），不双写。

**范围（Phase 1）**：只有**零正确性影响的观测类** knob 可热生效——目前是 access_log 家族。结构/连接类（`rdma` / `lib` / `register` / `node_dedup`）和 **C 客户端里 `static const` 缓存的** knob（`peer_cooldown` / `get_miss_retries` / `node_dedup_log` …）**仍需重启**：它们在 native 客户端构造期或首个 op 就冻结了，Python 侧翻转 env 无效，属单独的 native 改动（Phase 2）。

## 日志格式与各 op 的结果含义

`<op>(<参数>) : <结果> <耗时秒>`。覆盖的接口与对应 `<结果>`：

| op | 触发时机 | `<结果>` 取值 |
|---|---|---|
| `init` | 插件构造（打开连接 + MDS 发现） | `ok static` / `ok mds-discovery` |
| `batch_set_v1` | 零拷贝写（生产热路径） | `ok H/N`；MLA 非 0 号 rank 为 `backup_skip` |
| `batch_get_v1` | 零拷贝读（生产热路径） | `hits=H/N` |
| `batch_exists` | 前缀命中探测 | `prefix=N/total` |
| `batch_set_v2` / `batch_get_v2` | 多池（Mamba/SWA/DeepSeek-V4） | 逐池 `kv ok=3/3, extra ok=3/3` |
| `batch_exists_v2` | 多池前缀探测 | `kv=K/total <pool>=…` |
| `set` / `get` | generic 单页读写 | `ok`/`fail`/`fail none`；`hit`/`miss`，参数带字节数 |
| `batch_set` / `batch_get` | generic 批量（循环 set/get） | `ok H/N` / `hits=H/N` |
| `exists` | 单 key 存在性 | `found` / `not_found` |

> 异常会被记成 `... : FAIL <异常类型>: <信息>` 并**照常向上抛出**（不吞异常）。
> `init` 里若 `dfkv_open` / 发现启动失败即属此类。最早的 `interface_v1` 必填校验在日志配置之前就 `raise`，不记录。

## 性能

- **关闭时**：约几十纳秒/次。`access_log()` 返回一个冻结的 noop 单例，跳过计时与参数格式化。
- **开启时**：约几微秒/次。落盘走 `logging.handlers.QueueHandler` + 后台 `QueueListener` 线程，
  前台只入队、**不在热路径同步写/flush**。进程退出时 `atexit` 自动 flush。

## 验证 / 复跑

无需 GPU/torch，单测会拉起真实 `dfkv_server` 节点驱动插件：

```bash
DFKV_BUILD=$(pwd)/build python3 test/python/test_dfkv_hicache.py   # 含 access-log + 滚动用例
# 滚动用例不依赖 node/so，可单独快速跑：
python3 -m unittest test_dfkv_hicache.DfkvAccessLogRotationTest -v
# 热开关（纯 Python，无需 node/so/GPU）：
python3 -m unittest test_dfkv_hot_config -v
# vLLM/LMCache 连接器热开关 + vendored watcher 一致性：
python3 -m unittest test_connector_access_log_hot test_telemetry_vendor_sync -v
```

## 实现位置

- [integration/hicache/dfkv_access_log.py](../integration/hicache/dfkv_access_log.py) — `configure()`（启动基线，幂等）/ `apply_hot()`（热覆盖）/ `access_log()` 上下文管理器、noop 单例、异步队列、格式化辅助。
- [integration/hicache/dfkv_hot_config.py](../integration/hicache/dfkv_hot_config.py) — **canonical** 控制文件 watcher（后台守护线程）+ 通用 `register()` 注册表；opt-in（`DFKV_HOT_CONFIG` 设了才启动）。vLLM/LMCache 各 vendor 一份 byte-identical 的 `_hot_config.py`（`deploy/sync_telemetry.sh` 同步，`test/python/test_telemetry_vendor_sync.py` 守门）。
- vLLM/LMCache 连接器的 `access_log.py`（各自独立文件，机制同 hicache）+ `_hot_config.py`（vendored），接线在 `dfkv_client.py` / `native_client.py` 的 `__init__`（`configure`+`register`+`start`）与 `close()`（`stop`）。
- [integration/hicache/dfkv_hicache.py](../integration/hicache/dfkv_hicache.py) — `__init__` 里调 `configure()`、`register("access_log", apply_hot)` 并 `start()` watcher，并用 `with access_log(...)` 包住各继承接口。
- [test/python/test_dfkv_hicache.py](../test/python/test_dfkv_hicache.py) — `DfkvAccessLogTest`（接口逐 op 日志）+ `DfkvAccessLogRotationTest`（滚动/清理）测试类。
- [test/python/test_dfkv_hot_config.py](../test/python/test_dfkv_hot_config.py) — 热开关 + watcher 文件生命周期（纯 Python，无 native 依赖）。
