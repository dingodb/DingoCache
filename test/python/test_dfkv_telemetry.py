"""Unit tests for the shared dfkv_telemetry push-metrics layer.

GPU/OTel/Prometheus-free: exercises the in-process counter mirror and the
zero-cost-when-off path. (End-to-end OTLP push is validated in the M3 backend
stack, which needs the OTel SDK + a running Collector.)
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "integration", "hicache"))  # <repo>/integration/hicache

import dfkv_telemetry  # noqa: E402
from dfkv_telemetry import config, metrics, tracing  # noqa: E402

_ENV_KEYS = [
    config.ENV_METRICS_ENABLED, config.ENV_TELEMETRY_ENABLED,
    config.ENV_CONNECTOR_ID, config.ENV_OTLP_ENDPOINT,
    config.ENV_EXPORT_INTERVAL_MS,
]


class TelemetryTestBase(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in _ENV_KEYS}
        for k in _ENV_KEYS:
            os.environ.pop(k, None)
        metrics._reset_for_test()

    def tearDown(self):
        metrics._reset_for_test()
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class DisabledByDefaultTest(TelemetryTestBase):
    def test_off_by_default_is_noop(self):
        metrics.configure({}, connector_type=config.TYPE_VLLM, tp_rank=0)
        self.assertFalse(metrics.is_enabled())
        # op() returns the frozen no-op; setting attrs / raising is harmless.
        with metrics.op("put", num_keys=3, num_bytes=4096) as m:
            m.status = "fail"
            m.keys = 99
        self.assertEqual(metrics.snapshot(), {})

    def test_noop_does_not_swallow_exceptions(self):
        metrics.configure({}, connector_type=config.TYPE_LMCACHE)
        with self.assertRaises(ValueError):
            with metrics.op("get", num_keys=1):
                raise ValueError("boom")


class ConnectorIdTest(TelemetryTestBase):
    def test_default_id_is_host_pid_rank(self):
        cid = config.resolve_connector_id({}, tp_rank=2)
        self.assertEqual(cid.rsplit("_", 1)[1], "2")
        self.assertEqual(cid, "{}_{}_2".format(__import__("socket").gethostname(), os.getpid()))

    def test_default_id_passes_mds_isvalidgrouporid(self):
        # The auto-derived id is the etcd key tail /dfkv/v1/groups/<g>/clients/<id>,
        # so it MUST match the MDS IsValidGroupOrId alphabet [A-Za-z0-9._-]. A ":"
        # (an earlier default) is rejected -> UpsertClient returns kInvalid -> the
        # registrar never advances to the heartbeat loop -> dfkvctl clients is empty.
        cid = config.resolve_connector_id({}, tp_rank=3)
        self.assertTrue(cid and len(cid) <= 128)
        for c in cid:
            self.assertTrue(
                (c.isalnum() and c.isascii()) or c in "._-",
                f"connector_id {cid!r} has char {c!r} outside IsValidGroupOrId alphabet")

    def test_env_overrides_default(self):
        os.environ[config.ENV_CONNECTOR_ID] = "node-A-rank0"
        self.assertEqual(config.resolve_connector_id({}, tp_rank=7), "node-A-rank0")

    def test_extra_config_wins_over_env(self):
        os.environ[config.ENV_CONNECTOR_ID] = "from-env"
        self.assertEqual(
            config.resolve_connector_id({"connector_id": "from-cfg"}, tp_rank=0),
            "from-cfg")


class VersionLabelTest(TelemetryTestBase):
    """version / native_version ride on the recorder (and, with OTel, the
    resource) so the dashboard can spot connector vs libdfkv.so version skew."""

    def test_dist_version_unknown_is_empty_not_raises(self):
        # A distribution that surely isn't installed -> "" (never raises).
        self.assertEqual(config.dist_version("dfkv-nonexistent-xyz"), "")
        self.assertEqual(config.dist_version(""), "")

    def test_recorder_captures_versions_when_enabled(self):
        os.environ[config.ENV_METRICS_ENABLED] = "1"
        metrics.configure({}, connector_type=config.TYPE_VLLM, tp_rank=0,
                          version="2.0.0", native_version="1.6.3")
        self.assertTrue(metrics.is_enabled())
        self.assertEqual(metrics._recorder.version, "2.0.0")
        self.assertEqual(metrics._recorder.native_version, "1.6.3")

    def test_versions_default_empty(self):
        os.environ[config.ENV_METRICS_ENABLED] = "1"
        metrics.configure({}, connector_type=config.TYPE_VLLM)
        self.assertEqual(metrics._recorder.version, "")
        self.assertEqual(metrics._recorder.native_version, "")


class EnabledInProcessTest(TelemetryTestBase):
    """Enabled via env, OTel absent in this env => in-process-only counters."""

    def _enable(self, **cfg):
        os.environ[config.ENV_METRICS_ENABLED] = "1"
        metrics.configure(cfg, connector_type=config.TYPE_HICACHE, tp_rank=1,
                          model="m")

    def test_records_requests_keys_bytes_status(self):
        self._enable()
        self.assertTrue(metrics.is_enabled())
        with metrics.op("put", num_keys=3, num_bytes=4096) as m:
            m.status = "ok"
        with metrics.op("get", num_keys=5, num_bytes=8192):  # inferred ok
            pass
        snap = metrics.snapshot()
        self.assertEqual(snap["ops"]["put/ok"], 1)
        self.assertEqual(snap["ops"]["get/ok"], 1)
        self.assertEqual(snap["keys"]["put"], 3)
        self.assertEqual(snap["keys"]["get"], 5)
        self.assertEqual(snap["bytes"]["put"], 4096)
        self.assertEqual(snap["bytes"]["get"], 8192)
        self.assertEqual(snap["connector_type"], config.TYPE_HICACHE)

    def test_exception_marks_fail(self):
        self._enable()
        try:
            with metrics.op("put", num_keys=2, num_bytes=10):
                raise RuntimeError("io error")
        except RuntimeError:
            pass
        snap = metrics.snapshot()
        self.assertEqual(snap["ops"]["put/fail"], 1)
        self.assertEqual(snap["ops"].get("put/ok", 0), 0)

    def test_duration_sum_count_and_max_reset(self):
        self._enable()
        rec = metrics._recorder  # the real recorder
        with metrics.op("get", num_keys=1, num_bytes=1):
            pass
        snap = metrics.snapshot()
        self.assertEqual(snap["dur_cnt"]["get"], 1)
        self.assertGreaterEqual(snap["dur_sum"]["get"], 0.0)
        # take_max reads-and-resets (the observable gauge relies on this).
        first = rec.take_max("get")
        self.assertGreaterEqual(first, 0.0)
        self.assertEqual(rec.take_max("get"), 0.0)

    def test_umbrella_switch_enables(self):
        os.environ[config.ENV_TELEMETRY_ENABLED] = "true"
        metrics.configure({}, connector_type=config.TYPE_VLLM)
        self.assertTrue(metrics.is_enabled())

    def test_configure_is_idempotent(self):
        self._enable()
        # second configure (even disabling) must not replace the live recorder
        metrics.configure({"metrics": False}, connector_type=config.TYPE_VLLM)
        self.assertTrue(metrics.is_enabled())


    def test_lifecycle_reference_count_allows_reconfigure(self):
        self._enable()
        metrics.configure({}, connector_type=config.TYPE_VLLM)
        self.assertEqual(metrics._users, 2)
        metrics.release()
        self.assertTrue(metrics.is_enabled())
        metrics.release()
        self.assertEqual(metrics._users, 0)
        self.assertFalse(metrics.is_enabled())


class LiteralZeroOffTest(TelemetryTestBase):
    """When off, `if _m:` must be falsy so the connector never evaluates the
    (potentially O(batch)) metric args — literal zero overhead."""

    def test_disabled_op_is_falsy_and_skips_arg_eval(self):
        metrics.configure({}, connector_type=config.TYPE_VLLM)  # off by default
        calls = []

        def expensive():
            calls.append(1)
            return 4096

        with metrics.op("put", num_keys=3) as m:
            self.assertFalse(bool(m))
            if m:
                m.bytes = expensive()
        self.assertEqual(calls, [])           # never evaluated while off
        self.assertEqual(metrics.snapshot(), {})

    def test_enabled_op_is_truthy_and_records_guarded_bytes(self):
        os.environ[config.ENV_METRICS_ENABLED] = "1"
        metrics.configure({}, connector_type=config.TYPE_VLLM)
        calls = []

        def expensive():
            calls.append(1)
            return 4096

        with metrics.op("put", num_keys=3) as m:
            self.assertTrue(bool(m))
            if m:
                m.bytes = expensive()
        self.assertEqual(calls, [1])
        self.assertEqual(metrics.snapshot()["bytes"]["put"], 4096)


_SAMPLE_SNAPSHOT = """# HELP dfkv_client_peer_latency_seconds ...
# TYPE dfkv_client_peer_latency_seconds histogram
dfkv_client_peer_latency_seconds_bucket{le="0.0005",peer="10.0.0.1:9000"} 2
dfkv_client_peer_latency_seconds_sum{peer="10.0.0.1:9000"} 0.0030
dfkv_client_peer_latency_seconds_count{peer="10.0.0.1:9000"} 3
dfkv_client_peer_latency_max_seconds{peer="10.0.0.1:9000"} 0.0015
dfkv_client_peer_latency_seconds_sum{peer="10.0.0.2:9000"} 0.0100
dfkv_client_peer_latency_seconds_count{peer="10.0.0.2:9000"} 2
dfkv_client_peer_latency_max_seconds{peer="10.0.0.2:9000"} 0.0060
"""


class PeerLatencyTest(TelemetryTestBase):
    def test_parse_peer_latency(self):
        from dfkv_telemetry.metrics_push import parse_peer_latency
        d = parse_peer_latency(_SAMPLE_SNAPSHOT)
        self.assertEqual(d["10.0.0.1:9000"], {"sum": 0.003, "count": 3.0, "max": 0.0015})
        self.assertAlmostEqual(d["10.0.0.2:9000"]["max"], 0.006)
        self.assertEqual(set(d), {"10.0.0.1:9000", "10.0.0.2:9000"})

    def test_poller_computes_avg_and_max(self):
        from dfkv_telemetry.metrics_push import PeerLatencyPoller
        os.environ[config.ENV_METRICS_ENABLED] = "1"
        metrics.configure({}, connector_type=config.TYPE_VLLM)
        rec = metrics._recorder
        poller = PeerLatencyPoller(lambda: _SAMPLE_SNAPSHOT, rec, interval_s=0)
        res = poller.poll_once()
        self.assertAlmostEqual(res["10.0.0.1:9000"]["avg"], 0.001)
        self.assertAlmostEqual(res["10.0.0.1:9000"]["max"], 0.0015)
        self.assertAlmostEqual(res["10.0.0.2:9000"]["avg"], 0.005)
        # recorder state is updated (observable gauges read it)
        self.assertIn("10.0.0.1:9000", rec.peer_latency_snapshot())

    def test_poller_windowed_avg_uses_deltas(self):
        from dfkv_telemetry.metrics_push import PeerLatencyPoller
        os.environ[config.ENV_METRICS_ENABLED] = "1"
        metrics.configure({}, connector_type=config.TYPE_VLLM)
        texts = iter([
            "dfkv_client_peer_latency_seconds_sum{peer=\"p\"} 0.002\n"
            "dfkv_client_peer_latency_seconds_count{peer=\"p\"} 2\n"
            "dfkv_client_peer_latency_max_seconds{peer=\"p\"} 0.0015\n",
            "dfkv_client_peer_latency_seconds_sum{peer=\"p\"} 0.005\n"
            "dfkv_client_peer_latency_seconds_count{peer=\"p\"} 3\n"
            "dfkv_client_peer_latency_max_seconds{peer=\"p\"} 0.0030\n",
        ])
        poller = PeerLatencyPoller(lambda: next(texts), metrics._recorder, interval_s=0)
        poller.poll_once()                       # baseline
        res = poller.poll_once()                 # delta: (0.005-0.002)/(3-2) = 0.003
        self.assertAlmostEqual(res["p"]["avg"], 0.003)
        self.assertAlmostEqual(res["p"]["max"], 0.003)  # lifetime max

    def test_poll_failure_preserves_data_and_marks_snapshot_stale(self):
        from dfkv_telemetry.metrics_push import PeerLatencyPoller
        os.environ[config.ENV_METRICS_ENABLED] = "1"
        metrics.configure({}, connector_type=config.TYPE_VLLM)
        rec = metrics._recorder
        poller = PeerLatencyPoller(lambda: _SAMPLE_SNAPSHOT, rec, interval_s=0)
        poller.poll_once()
        before = rec.peer_latency_snapshot()
        poller._get_text = lambda: ""
        with self.assertRaisesRegex(ValueError, "empty or invalid"):
            poller.poll_once()
        self.assertEqual(rec.peer_latency_snapshot(), before)
        health = rec.client_poll_snapshot()
        self.assertEqual(health["success"], 0.0)
        self.assertEqual(health["errors"], 1)
        self.assertGreater(health["last_success_unixtime"], 0.0)


class StdlibExporterTest(TelemetryTestBase):
    """The default exporter is the pure-stdlib OTLP/HTTP-JSON pusher (no OTel SDK)."""

    def _enabled(self):
        os.environ[config.ENV_METRICS_ENABLED] = "1"
        metrics.configure({}, connector_type=config.TYPE_HICACHE, tp_rank=2)
        return metrics._recorder

    def test_default_exporter_is_stdlib_not_otel(self):
        rec = self._enabled()
        self.assertTrue(metrics.is_enabled())
        self.assertIsNotNone(rec._stdlib)   # stdlib pusher started
        self.assertIsNone(rec._otel)        # OTel SDK not used

    def test_stdlib_warns_on_standard_grpc_port(self):
        os.environ[config.ENV_METRICS_ENABLED] = "1"
        os.environ[config.ENV_OTLP_ENDPOINT] = "http://collector:4317"
        with self.assertWarnsRegex(RuntimeWarning, "OTLP/HTTP JSON"):
            metrics.configure({}, connector_type=config.TYPE_HICACHE, tp_rank=2)

    def test_build_payload_structure_and_values(self):
        rec = self._enabled()
        with metrics.op("put", num_keys=3) as m:
            if m:
                m.bytes = 4096
        with metrics.op("get", num_keys=5) as m:
            if m:
                m.bytes = 8192
        try:
            with metrics.op("put", num_keys=1) as m:
                raise RuntimeError("io")
        except RuntimeError:
            pass
        rm = rec._stdlib.build()["resourceMetrics"][0]
        ra = {a["key"]: a["value"] for a in rm["resource"]["attributes"]}
        self.assertEqual(ra["dfkv.connector_type"]["stringValue"], "hicache")
        self.assertIn("dfkv.connector_id", ra)
        by_name = {m["name"]: m for m in rm["scopeMetrics"][0]["metrics"]}
        for n in ("dfkv_connector_ops_total", "dfkv_connector_keys_total",
                  "dfkv_connector_bytes_total", "dfkv_connector_op_seconds",
                  "dfkv_connector_info"):
            self.assertIn(n, by_name)
        ops = {}
        for dp in by_name["dfkv_connector_ops_total"]["sum"]["dataPoints"]:
            a = {x["key"]: x["value"]["stringValue"] for x in dp["attributes"]}
            ops[(a["op"], a["status"])] = int(dp["asInt"])
        self.assertEqual(ops[("put", "ok")], 1)
        self.assertEqual(ops[("put", "fail")], 1)
        self.assertEqual(ops[("get", "ok")], 1)
        # histogram: bucketCounts length == bounds + 1 (overflow)
        hist = by_name["dfkv_connector_op_seconds"]["histogram"]["dataPoints"][0]
        self.assertEqual(len(hist["bucketCounts"]), len(hist["explicitBounds"]) + 1)

    def test_metrics_url_normalization(self):
        from dfkv_telemetry.otlp_json import metrics_url
        self.assertEqual(metrics_url("http://h:4318"), "http://h:4318/v1/metrics")
        self.assertEqual(metrics_url("http://h:4318/"), "http://h:4318/v1/metrics")
        self.assertEqual(metrics_url("h:4318"), "http://h:4318/v1/metrics")
        self.assertEqual(metrics_url("http://h:4318/v1/metrics"), "http://h:4318/v1/metrics")
        self.assertEqual(metrics_url("http://h:4318/v1/metrics/"), "http://h:4318/v1/metrics")
        self.assertEqual(metrics_url(""), "http://localhost:4318/v1/metrics")
        self.assertEqual(
            metrics_url("http://h:4318/v1/traces"),
            "http://h:4318/v1/metrics")


    def test_background_push_warns_once_and_reports_recovery(self):
        import warnings
        from dfkv_telemetry.otlp_json import StdlibExporter

        exp = StdlibExporter(self._enabled(), "http://127.0.0.1:1", interval_s=0)

        def fail():
            raise OSError("collector down")

        exp.push_once = fail
        with self.assertWarnsRegex(RuntimeWarning, "metrics export failed"):
            self.assertFalse(exp._push_guarded())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertFalse(exp._push_guarded())
        self.assertEqual(caught, [])
        exp.push_once = lambda: 200
        with self.assertWarnsRegex(RuntimeWarning, "export recovered"):
            self.assertTrue(exp._push_guarded())

    def test_stop_performs_one_final_push(self):
        from dfkv_telemetry.otlp_json import StdlibExporter
        exp = StdlibExporter(
            self._enabled(), "http://127.0.0.1:1", interval_s=0)
        calls = []
        exp._push_guarded = lambda: calls.append("push") or True
        exp.stop()
        self.assertEqual(calls, ["push"])

    def test_failed_push_restores_window_maximum(self):
        from dfkv_telemetry.otlp_json import StdlibExporter
        rec = self._enabled()
        with metrics.op("get", num_keys=1, num_bytes=1):
            pass
        exp = StdlibExporter(rec, "http://127.0.0.1:1", interval_s=0)

        class Down:
            def open(self, *_args, **_kwargs):
                raise OSError("collector down")

        exp._opener = Down()
        with self.assertRaisesRegex(OSError, "collector down"):
            exp.push_once()
        self.assertGreater(rec.export_snapshot()["dur_max"]["get"], 0.0)


    def test_push_once_to_local_http_server(self):
        import http.server
        import json as _json
        import socketserver
        import threading
        captured = {}

        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                captured["body"] = self.rfile.read(n)
                captured["ctype"] = self.headers.get("Content-Type")
                self.send_response(200)
                self.end_headers()

            def log_message(self, *a):
                pass

        srv = socketserver.TCPServer(("127.0.0.1", 0), H)
        port = srv.server_address[1]
        threading.Thread(target=srv.handle_request, daemon=True).start()

        rec = self._enabled()
        with metrics.op("put", num_keys=2) as m:
            if m:
                m.bytes = 100
        from dfkv_telemetry.otlp_json import StdlibExporter
        exp = StdlibExporter(rec, "http://127.0.0.1:%d" % port, interval_s=0)
        status = exp.push_once()
        srv.server_close()
        self.assertEqual(status, 200)
        self.assertEqual(captured["ctype"], "application/json")
        body = _json.loads(captured["body"].decode())
        names = [m["name"] for m in body["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]]
        self.assertIn("dfkv_connector_ops_total", names)


try:
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader  # noqa: F401
    _HAVE_OTEL = True
except Exception:
    _HAVE_OTEL = False


@unittest.skipUnless(_HAVE_OTEL, "opentelemetry SDK not installed")
class OtelPushTest(TelemetryTestBase):
    """Validates the real OTel instrument output via an in-memory reader (no
    network). Skipped when the SDK is absent; exercised in CI images with the
    'otel' extra and in the M3 backend stack end-to-end."""

    def _collect(self, reader):
        out = {}
        res = {}
        data = reader.get_metrics_data()
        for rm in data.resource_metrics:
            res = dict(rm.resource.attributes)
            for sm in rm.scope_metrics:
                for metric in sm.metrics:
                    pts = []
                    for dp in metric.data.data_points:
                        attrs = dict(dp.attributes)
                        if hasattr(dp, "value"):
                            pts.append((attrs, dp.value))
                        else:
                            pts.append((attrs, ("count", dp.count)))
                    out[metric.name] = pts
        return out, res

    def test_otlp_instruments_emit_expected_series(self):
        from opentelemetry.sdk.metrics.export import InMemoryMetricReader
        os.environ[config.ENV_METRICS_ENABLED] = "1"
        os.environ[config.ENV_CONNECTOR_ID] = "node-A-rank0"
        reader = InMemoryMetricReader()
        metrics.configure({}, connector_type=config.TYPE_VLLM, tp_rank=0,
                          model="m", version="2.0.0", native_version="1.6.3",
                          _readers=[reader])
        self.assertTrue(metrics.is_enabled())
        with metrics.op("put", num_keys=3, num_bytes=4096) as m:
            m.status = "ok"
        with metrics.op("get", num_keys=5, num_bytes=8192):
            pass
        try:
            with metrics.op("put", num_keys=2, num_bytes=10):
                raise RuntimeError("io")
        except RuntimeError:
            pass

        native = (
            'dfkv_client_op_requests_total{op="get"} 8\n'
            'dfkv_client_op_keys_total{op="get"} 64\n'
            'dfkv_client_op_max_seconds{op="get"} 0.02\n'
            'dfkv_client_op_latency_seconds_bucket{le="0.1",op="get"} 8\n'
            'dfkv_client_op_latency_seconds_sum{op="get"} 0.08\n'
            'dfkv_client_op_latency_seconds_count{op="get"} 8\n'
            'dfkv_client_dedup_hits_total 7\n'
            'dfkv_client_ring_members 1\n'
            'dfkv_transport_pool_connections 4\n'
        )
        rec = metrics._recorder
        rec.update_client_ops(metrics.parse_client_ops(native))
        rec.update_client_dedup(metrics.parse_client_dedup(native))
        rec.update_client_status(metrics.parse_client_status(native))
        rec.update_peer_latency({
            "node:28101": {
                "avg": 0.001,
                "max": 0.002,
                "sum": 0.008,
                "count": 8,
            },
        })
        rec.record_client_poll(True)

        found, res = self._collect(reader)
        # identity rides on resource attributes (-> Prometheus labels via the
        # Collector's resource_to_telemetry_conversion).
        self.assertEqual(res.get("dfkv.connector_id"), "node-A-rank0")
        self.assertEqual(res.get("dfkv.connector_type"), config.TYPE_VLLM)
        # version labels (-> dfkv_version / dfkv_native_version in Prometheus)
        self.assertEqual(res.get("dfkv.version"), "2.0.0")
        self.assertEqual(res.get("dfkv.native_version"), "1.6.3")
        ops = {tuple(sorted(a.items())): v for a, v in found["dfkv_connector_ops_total"]}
        self.assertEqual(ops[(("op", "put"), ("status", "ok"))], 1)
        self.assertEqual(ops[(("op", "put"), ("status", "fail"))], 1)
        self.assertEqual(ops[(("op", "get"), ("status", "ok"))], 1)
        keys = {a["op"]: v for a, v in found["dfkv_connector_keys_total"]}
        self.assertEqual(keys["put"], 5)  # 3 + 2
        self.assertEqual(keys["get"], 5)
        nbytes = {a["op"]: v for a, v in found["dfkv_connector_bytes_total"]}
        self.assertEqual(nbytes["put"], 4106)
        self.assertEqual(nbytes["get"], 8192)
        self.assertIn("dfkv_connector_op_seconds", found)
        self.assertIn("dfkv_connector_op_max_seconds", found)
        self.assertTrue(any(v == 1 for _, v in found["dfkv_connector_info"]))
        for name in (
                "dfkv_connector_op_requests_total",
                "dfkv_connector_op_latency_seconds_bucket",
                "dfkv_connector_op_latency_seconds_sum",
                "dfkv_connector_op_latency_seconds_count",
                "dfkv_connector_op_lifetime_max_seconds",
                "dfkv_connector_dedup_hits_total",
                "dfkv_connector_ring_members",
                "dfkv_connector_transport_pool_connections",
                "dfkv_client_peer_latency_seconds_sum",
                "dfkv_client_peer_latency_seconds_count",
                "dfkv_connector_client_stats_poll_success",
                "dfkv_connector_client_stats_last_success_unixtime",
                "dfkv_connector_client_stats_poll_errors_total"):
            self.assertIn(name, found)
        metrics.shutdown()


class ClientOpForwardTest(TelemetryTestBase):
    """Per-op metrics are computed once in the C++ KVClient (dfkv_client_op_*);
    the snapshot poller parses them and re-emits dfkv_connector_op_* with this
    connector's identity, so all connectors report their full request mix."""

    SNAP = (
        'dfkv_client_op_requests_total{op="put"} 10\n'
        'dfkv_client_op_keys_total{op="put"} 640\n'
        'dfkv_client_op_hits_total{op="put"} 600\n'        # 40 pages failed to write
        'dfkv_client_op_bytes_total{op="put"} 2621440\n'
        'dfkv_client_op_max_seconds{op="put"} 0.020000\n'
        'dfkv_client_op_latency_seconds_bucket{le="0.001",op="put"} 2\n'
        'dfkv_client_op_latency_seconds_bucket{le="0.1",op="put"} 10\n'
        'dfkv_client_op_latency_seconds_bucket{le="+Inf",op="put"} 10\n'
        'dfkv_client_op_latency_seconds_sum{op="put"} 0.08\n'
        'dfkv_client_op_latency_seconds_count{op="put"} 10\n'
        'dfkv_client_op_requests_total{op="exist"} 5\n'
        'dfkv_client_op_keys_total{op="exist"} 6960\n'
        'dfkv_client_op_hits_total{op="exist"} 6960\n'
        'dfkv_client_op_max_seconds{op="exist"} 48.500000\n'
        'dfkv_client_op_latency_seconds_bucket{le="0.1",op="exist"} 4\n'
        'dfkv_client_op_latency_seconds_bucket{le="+Inf",op="exist"} 5\n'  # 1 in +Inf (48s)
        'dfkv_client_op_latency_seconds_sum{op="exist"} 48.6\n'
        'dfkv_client_op_latency_seconds_count{op="exist"} 5\n'
    )

    DEDUP_SNAP = (
        'dfkv_client_dedup_hits_total 12000\n'
        'dfkv_client_dedup_fetches_total 1600\n'
        'dfkv_client_dedup_wait_hits_total 10400\n'
        'dfkv_client_dedup_wait_timeouts_total 8\n'
        'dfkv_client_gpu_dedup_hits_total 500\n'
        'dfkv_client_op_requests_total{op="get"} 3\n'  # op-labeled: must NOT be a dedup counter
    )

    def test_parse_client_ops(self):
        d = metrics.parse_client_ops(self.SNAP)
        self.assertEqual(d["put"]["keys"], 640.0)
        self.assertEqual(d["put"]["hits"], 600.0)   # keys-hits = failed writes
        self.assertEqual(d["exist"]["max"], 48.5)    # the batch_exist tail
        self.assertEqual(len(d["put"]["buckets"]), 3)

    def test_parse_client_dedup(self):
        d = metrics.parse_client_dedup(self.DEDUP_SNAP)
        self.assertEqual(d["dfkv_client_dedup_hits_total"], 12000.0)
        self.assertEqual(d["dfkv_client_dedup_wait_timeouts_total"], 8.0)
        self.assertEqual(d["dfkv_client_gpu_dedup_hits_total"], 500.0)
        # op-labeled lines must be excluded (they belong to parse_client_ops)
        self.assertNotIn("dfkv_client_op_requests_total", d)

    def test_dedup_forwarded_to_otlp(self):
        from dfkv_telemetry import otlp_json
        os.environ[config.ENV_METRICS_ENABLED] = "1"
        metrics.configure({}, connector_type=config.TYPE_HICACHE, tp_rank=0, model="m")
        rec = metrics._recorder
        rec.update_client_dedup(metrics.parse_client_dedup(self.DEDUP_SNAP))
        payload = otlp_json.build_payload(rec, rec.export_snapshot(), 1, 2)
        names = {m["name"] for m in
                 payload["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]}
        self.assertIn("dfkv_connector_dedup_hits_total", names)
        self.assertIn("dfkv_connector_gpu_dedup_hits_total", names)

    def test_forward_to_otlp_with_identity(self):
        from dfkv_telemetry import otlp_json
        os.environ[config.ENV_METRICS_ENABLED] = "1"
        metrics.configure({}, connector_type=config.TYPE_HICACHE, tp_rank=2, model="m")
        rec = metrics._recorder
        rec.update_client_ops(metrics.parse_client_ops(self.SNAP))
        payload = otlp_json.build_payload(rec, rec.export_snapshot(), 1, 2)
        ms = {m["name"]: m for m in
              payload["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]}
        for name in (
                "dfkv_connector_op_requests_total",
                "dfkv_connector_op_keys_total",
                "dfkv_connector_op_hits_total",
                "dfkv_connector_op_bytes_total",
                "dfkv_connector_op_lifetime_max_seconds",
                "dfkv_connector_op_latency_seconds"):
            self.assertIn(name, ms)
        metric_names = [
            metric["name"]
            for metric in
            payload["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]
        ]
        self.assertEqual(len(metric_names), len(set(metric_names)))
        attrs = {a["key"] for a in
                 payload["resourceMetrics"][0]["resource"]["attributes"]}
        self.assertIn("dfkv.connector_type", attrs)
        self.assertIn("dfkv.connector_id", attrs)
        # the exist histogram preserved the 48s op in the +Inf bucket
        ehist = [dp for dp in ms["dfkv_connector_op_latency_seconds"]["histogram"]["dataPoints"]
                 if dp["attributes"][0]["value"]["stringValue"] == "exist"][0]
        self.assertEqual(ehist["count"], "5")
        self.assertEqual(ehist["bucketCounts"][-1], "1")  # +Inf bucket = the 48s op
        metrics.shutdown()


class NativeSnapshotDeltaTest(unittest.TestCase):
    def test_uncommitted_counter_delta_is_replayed(self):
        from dfkv_common.client_metrics import SnapshotDelta
        delta = SnapshotDelta()
        text = "dfkv_client_io_errors_total 5\n"
        first = delta.prepare(text)
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].value, 5.0)
        replay = delta.prepare(text)
        self.assertEqual(replay[0].value, 5.0)
        delta.commit(replay[0])
        self.assertEqual(delta.prepare(text), [])


class ClientStatusForwardTest(TelemetryTestBase):
    SNAP = (
        'dfkv_client_transport_info{transport="rdma"} 1\n'
        'dfkv_client_ring_members 0\n'
        'dfkv_client_mds_reachable 0\n'
        'dfkv_client_io_errors_total 2\n'
        'dfkv_client_peer_errors_total{peer="unbounded"} 9\n'
        'dfkv_rdma_client_rail_errors_total{dev="ib0"} 3\n'
        'dfkv_rdma_client_rail_errors_total{dev="ib1"} 0\n'
        'dfkv_rdma_client_mr_registered_bytes 134217728\n'
        'dfkv_rdma_cq_completions_total 11\n'
        'dfkv_transport_pool_connections 8\n'
        'dfkv_transport_pool_retirements_total{reason="idle"} 2\n'
        'dfkv_client_op_requests_total{op="get"} 8\n'
        'dfkv_client_dedup_hits_total 4\n'
        'unknown_metric 99\n'
    )

    def test_parse_is_bounded_and_excludes_dedicated_families(self):
        samples = metrics.parse_client_status(self.SNAP)
        by_name = {sample["name"] for sample in samples}
        self.assertIn("dfkv_connector_ring_members", by_name)
        self.assertIn("dfkv_connector_mds_reachable", by_name)
        self.assertIn("dfkv_connector_rdma_rail_errors_total", by_name)
        self.assertIn("dfkv_connector_rdma_mr_registered_bytes", by_name)
        self.assertIn("dfkv_connector_rdma_cq_completions_total", by_name)
        self.assertIn("dfkv_connector_transport_pool_connections", by_name)
        self.assertIn("dfkv_connector_transport_pool_retirements_total", by_name)
        self.assertNotIn("dfkv_connector_peer_errors_total", by_name)
        self.assertNotIn("dfkv_connector_op_requests_total", by_name)
        self.assertNotIn("dfkv_connector_dedup_hits_total", by_name)

    def test_status_counters_and_zero_gauges_forward_to_otlp(self):
        from dfkv_telemetry import otlp_json
        os.environ[config.ENV_METRICS_ENABLED] = "1"
        metrics.configure({}, connector_type=config.TYPE_LMCACHE, tp_rank=3)
        rec = metrics._recorder
        poller = metrics.PeerLatencyPoller(
            lambda: self.SNAP, rec, interval_s=0)
        poller.poll_once()
        payload = otlp_json.build_payload(rec, rec.export_snapshot(), 1, 2)
        families = {
            metric["name"]: metric
            for metric in payload["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]
        }
        ring = families["dfkv_connector_ring_members"]
        self.assertEqual(ring["gauge"]["dataPoints"][0]["asDouble"], 0.0)
        io_errors = families["dfkv_connector_io_errors_total"]
        self.assertEqual(io_errors["sum"]["dataPoints"][0]["asInt"], "2")
        rails = families["dfkv_connector_rdma_rail_errors_total"]["sum"]["dataPoints"]
        self.assertEqual(len(rails), 2)
        values = {
            point["attributes"][0]["value"]["stringValue"]: point["asInt"]
            for point in rails
        }
        self.assertEqual(values, {"ib0": "3", "ib1": "0"})
        pool = families["dfkv_connector_transport_pool_connections"]
        self.assertEqual(pool["gauge"]["dataPoints"][0]["asDouble"], 8.0)
        self.assertEqual(
            families["dfkv_connector_client_stats_poll_success"]
            ["gauge"]["dataPoints"][0]["asDouble"], 1.0)
        self.assertIn(
            "dfkv_connector_client_stats_last_success_unixtime", families)
        poll_errors = families[
            "dfkv_connector_client_stats_poll_errors_total"]
        self.assertEqual(
            poll_errors["sum"]["dataPoints"][0]["asInt"], "0")
        metrics.shutdown()


# ---------------------------------------------------------------------------
# Tracing (connector-side spans: slow / sampled / failed, pushed over /v1/traces)
# ---------------------------------------------------------------------------

_TRACE_ENV_KEYS = [
    config.ENV_TRACING_ENABLED, config.ENV_TELEMETRY_ENABLED,
    config.ENV_TRACE_SLOW_REQUEST_MS, config.ENV_TRACE_SAMPLE_PERCENT,
    config.ENV_TRACE_EXPORT_INTERVAL_MS, config.ENV_TRACE_MAX_BUFFERED_SPANS,
    config.ENV_CONNECTOR_ID, config.ENV_OTLP_ENDPOINT,
]


class _CollectExporter:
    """Test span sink injected via tracing.configure(_exporter=...): collects
    enqueued spans in memory, no thread, no network."""

    def __init__(self):
        self.spans = []

    def enqueue(self, span):
        self.spans.append(span)

    def start(self):
        pass

    def stop(self):
        pass


class TracingTestBase(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in _TRACE_ENV_KEYS}
        for k in _TRACE_ENV_KEYS:
            os.environ.pop(k, None)
        tracing._reset_for_test()

    def tearDown(self):
        tracing._reset_for_test()
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _enable(self, exporter=None, **cfg):
        os.environ[config.ENV_TRACING_ENABLED] = "1"
        exp = exporter if exporter is not None else _CollectExporter()
        tracing.configure(cfg, connector_type=config.TYPE_HICACHE, tp_rank=2,
                          _exporter=exp)
        return exp


class TracingDisabledTest(TracingTestBase):
    def test_off_by_default_is_noop(self):
        tracing.configure({}, connector_type=config.TYPE_VLLM)
        self.assertFalse(tracing.is_enabled())
        with tracing.span("get", num_keys=3) as sp:
            self.assertFalse(bool(sp))   # falsy: callers can skip expensive attrs
            sp.hits = 3                  # dropped, never raises

    def test_noop_does_not_swallow_exceptions(self):
        tracing.configure({}, connector_type=config.TYPE_LMCACHE)
        with self.assertRaises(ValueError):
            with tracing.span("get", num_keys=1):
                raise ValueError("boom")

    def test_umbrella_switch_enables(self):
        os.environ[config.ENV_TELEMETRY_ENABLED] = "true"
        tracing.configure({}, connector_type=config.TYPE_VLLM, _exporter=_CollectExporter())
        self.assertTrue(tracing.is_enabled())

    def test_explicit_tracing_off_beats_umbrella_on(self):
        os.environ[config.ENV_TELEMETRY_ENABLED] = "1"
        tracing.configure({"tracing": False}, connector_type=config.TYPE_VLLM)
        self.assertFalse(tracing.is_enabled())


class TracingSampleDecisionTest(TracingTestBase):
    def test_slow_threshold(self):
        self._enable(trace_slow_request_ms=100, trace_sample_percent=0)
        t = tracing._tracer
        self.assertTrue(t.should_sample(150.0, False))    # over threshold
        self.assertTrue(t.should_sample(100.0, False))    # exactly at threshold
        self.assertFalse(t.should_sample(50.0, False))    # under, not sampled

    def test_failed_always_sampled(self):
        self._enable(trace_slow_request_ms=0, trace_sample_percent=0)
        self.assertTrue(tracing._tracer.should_sample(0.1, True))

    def test_percent_full_and_zero(self):
        self._enable(trace_slow_request_ms=0, trace_sample_percent=100)
        self.assertTrue(tracing._tracer.should_sample(0.1, False))
        tracing._reset_for_test()
        self._enable(trace_slow_request_ms=0, trace_sample_percent=0)
        self.assertFalse(tracing._tracer.should_sample(0.1, False))

    def test_percent_clamped_and_slow_disabled_by_zero(self):
        self._enable(trace_slow_request_ms=0, trace_sample_percent=999)
        # clamped to 100 -> always sample; slow_ms=0 -> latency never triggers alone
        self.assertEqual(tracing._tracer.sample_percent, 100.0)
        self.assertEqual(tracing._tracer.slow_ms, 0.0)


class TracingEmitTest(TracingTestBase):
    def test_slow_request_emits_span_with_attrs(self):
        # tiny slow threshold => every real op counts as slow and is emitted
        exp = self._enable(trace_slow_request_ms=0.0001, trace_sample_percent=0)
        with tracing.span("batch_get_v1", num_keys=3) as sp:
            self.assertTrue(bool(sp))
            sp.hits = 2
            sp.bytes = 4096
        self.assertEqual(len(exp.spans), 1)
        s = exp.spans[0]
        self.assertEqual(s["name"], "batch_get_v1")
        self.assertEqual(len(s["traceId"]), 32)   # 16-byte trace id
        self.assertEqual(len(s["spanId"]), 16)     # 8-byte span id
        self.assertNotIn("status", s)              # ok span leaves OTLP status unset
        a = {x["key"]: x["value"] for x in s["attributes"]}
        self.assertEqual(a["op"]["stringValue"], "batch_get_v1")
        self.assertEqual(a["status"]["stringValue"], "ok")
        self.assertEqual(a["dfkv.keys"]["intValue"], "3")
        self.assertEqual(a["dfkv.hits"]["intValue"], "2")
        self.assertEqual(a["dfkv.bytes"]["intValue"], "4096")

    def test_failed_op_traced_even_when_not_slow_or_sampled(self):
        exp = self._enable(trace_slow_request_ms=0, trace_sample_percent=0)
        with self.assertRaises(RuntimeError):
            with tracing.span("batch_set_v1", num_keys=2):
                raise RuntimeError("io error")
        self.assertEqual(len(exp.spans), 1)
        s = exp.spans[0]
        self.assertEqual(s["status"]["code"], 2)   # OTLP STATUS_ERROR
        self.assertIn("io error", s["status"]["message"])
        a = {x["key"]: x["value"] for x in s["attributes"]}
        self.assertEqual(a["status"]["stringValue"], "fail")
        self.assertIn("dfkv.error", a)

    def test_status_fail_set_by_caller_is_traced(self):
        exp = self._enable(trace_slow_request_ms=0, trace_sample_percent=0)
        with tracing.span("get", num_keys=1) as sp:
            sp.status = "fail"     # no exception, but caller flags failure
        self.assertEqual(len(exp.spans), 1)
        a = {x["key"]: x["value"] for x in exp.spans[0]["attributes"]}
        self.assertEqual(a["status"]["stringValue"], "fail")

    def test_fast_unsampled_op_emits_nothing(self):
        exp = self._enable(trace_slow_request_ms=0, trace_sample_percent=0)
        with tracing.span("get", num_keys=1):
            pass
        self.assertEqual(exp.spans, [])

    def test_configure_is_idempotent(self):
        exp = self._enable(trace_slow_request_ms=0.0001)
        tracing.configure({"tracing": False}, connector_type=config.TYPE_VLLM)
        self.assertTrue(tracing.is_enabled())   # second configure ignored


class _FakeIdentity:
    connector_type = "hicache"
    connector_id = "host:123:0"
    tp_rank = 0
    pid = 123
    version = "1.6.6"
    native_version = "1.6.6"


class OtlpTracesTest(unittest.TestCase):
    def test_traces_url_normalization(self):
        from dfkv_telemetry.otlp_traces import traces_url
        self.assertEqual(traces_url("http://h:4318"), "http://h:4318/v1/traces")
        self.assertEqual(traces_url("http://h:4318/"), "http://h:4318/v1/traces")
        self.assertEqual(traces_url("h:4318"), "http://h:4318/v1/traces")
        self.assertEqual(traces_url("http://h:4318/v1/traces"), "http://h:4318/v1/traces")
        self.assertEqual(traces_url("http://h:4318/v1/traces/"), "http://h:4318/v1/traces")
        self.assertEqual(traces_url(""), "http://localhost:4318/v1/traces")

    def test_make_span_and_build_payload(self):
        from dfkv_telemetry import otlp_traces
        sp = otlp_traces.make_span("get", "a" * 32, "b" * 16, 1000, 3000,
                                   {"op": "get", "dfkv.keys": 4}, failed=False)
        self.assertEqual(sp["kind"], 3)            # SPAN_KIND_CLIENT
        self.assertEqual(sp["startTimeUnixNano"], "1000")
        self.assertEqual(sp["endTimeUnixNano"], "3000")
        self.assertNotIn("status", sp)
        payload = otlp_traces.build_payload(_FakeIdentity(), [sp])
        rs = payload["resourceSpans"][0]
        ra = {a["key"]: a["value"] for a in rs["resource"]["attributes"]}
        self.assertEqual(ra["dfkv.connector_type"]["stringValue"], "hicache")
        self.assertEqual(ra["dfkv.connector_id"]["stringValue"], "host:123:0")
        self.assertEqual(ra["dfkv.version"]["stringValue"], "1.6.6")
        self.assertEqual(ra["service.name"]["stringValue"], "dfkv-hicache-connector")
        spans = rs["scopeSpans"][0]["spans"]
        self.assertEqual(spans[0]["name"], "get")

    def test_make_span_failed_sets_error_status(self):
        from dfkv_telemetry import otlp_traces
        sp = otlp_traces.make_span("put", "a" * 32, "b" * 16, 1, 2, {},
                                   failed=True, error="OSError: disk full")
        self.assertEqual(sp["status"]["code"], otlp_traces.STATUS_ERROR)
        self.assertEqual(sp["status"]["message"], "OSError: disk full")

    def test_exporter_buffer_drops_oldest_when_full(self):
        from dfkv_telemetry import otlp_traces
        exp = otlp_traces.StdlibSpanExporter(
            _FakeIdentity(), "http://127.0.0.1:1", interval_s=0, max_spans=2)
        for i in range(5):
            exp.enqueue({"i": i})
        self.assertEqual(exp.buffered(), 2)
        self.assertEqual(exp.dropped(), 3)       # 5 enqueued, cap 2 -> 3 dropped

    def test_flush_empty_is_noop(self):
        from dfkv_telemetry import otlp_traces
        exp = otlp_traces.StdlibSpanExporter(_FakeIdentity(), "http://127.0.0.1:1",
                                             interval_s=0)
        self.assertIsNone(exp.flush())

    def test_stop_performs_final_trace_flush(self):
        from dfkv_telemetry import otlp_traces
        exp = otlp_traces.StdlibSpanExporter(
            _FakeIdentity(), "http://127.0.0.1:1",
            interval_s=0)
        calls = []
        exp._flush_guarded = lambda: calls.append("flush") or True
        exp.stop()
        self.assertEqual(calls, ["flush"])


    def test_failed_flush_retains_spans_and_warns_once(self):
        import warnings
        from dfkv_telemetry import otlp_traces

        exp = otlp_traces.StdlibSpanExporter(
            _FakeIdentity(), "http://127.0.0.1:1", interval_s=0)
        class Down:
            def open(self, *_args, **_kwargs):
                raise OSError("collector down")

        exp._opener = Down()
        exp.enqueue({"traceId": "a" * 32, "spanId": "b" * 16,
                     "name": "get", "startTimeUnixNano": "1",
                     "endTimeUnixNano": "2", "kind": 3, "attributes": []})
        with self.assertWarnsRegex(RuntimeWarning, "trace export failed"):
            self.assertFalse(exp._flush_guarded())
        self.assertEqual(exp.buffered(), 1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertFalse(exp._flush_guarded())
        self.assertEqual(caught, [])
        exp.flush = lambda: 200
        with self.assertWarnsRegex(RuntimeWarning, "export recovered"):
            self.assertTrue(exp._flush_guarded())

    def test_flush_to_local_http_server(self):
        import http.server
        import json as _json
        import socketserver
        import threading
        from dfkv_telemetry import otlp_traces
        captured = {}

        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                captured["body"] = self.rfile.read(n)
                captured["ctype"] = self.headers.get("Content-Type")
                self.send_response(200)
                self.end_headers()

            def log_message(self, *a):
                pass

        srv = socketserver.TCPServer(("127.0.0.1", 0), H)
        port = srv.server_address[1]
        threading.Thread(target=srv.handle_request, daemon=True).start()

        exp = otlp_traces.StdlibSpanExporter(
            _FakeIdentity(), "http://127.0.0.1:%d" % port, interval_s=0)
        exp.enqueue(otlp_traces.make_span("get", "a" * 32, "b" * 16, 1, 2,
                                          {"op": "get"}))
        status = exp.flush()
        self.assertEqual(
            otlp_traces.traces_url("http://h:4318/v1/metrics"),
            "http://h:4318/v1/traces")
        srv.server_close()
        self.assertEqual(status, 200)
        self.assertEqual(captured["ctype"], "application/json")
        body = _json.loads(captured["body"].decode())
        self.assertIn("resourceSpans", body)
        self.assertEqual(exp.buffered(), 0)      # drained after flush


class RequiredConfigValidationTest(unittest.TestCase):
    """Startup validation of required connector config (isolation identity +
    ring endpoint). Pure config logic — no server / libdfkv.so."""

    def test_isolation_name_empty_raises(self):
        for bad in ("", "   ", None, 123):
            with self.assertRaises(config.DfkvConfigError, msg=repr(bad)):
                config.require_isolation_name(bad)

    def test_isolation_name_preserved_losslessly(self):
        identity = "org/model@rev:2026-08-02"
        self.assertEqual(config.require_isolation_name(identity), identity)

    # --- ring endpoint ------------------------------------------------------
    def test_ring_endpoint_requires_members_or_mds(self):
        with self.assertRaises(config.DfkvConfigError):
            config.require_ring_endpoint("", "")
        with self.assertRaises(config.DfkvConfigError):
            config.require_ring_endpoint(None, None)

    def test_ring_endpoint_ok_with_either(self):
        config.require_ring_endpoint("n0=127.0.0.1:1", "")   # no raise
        config.require_ring_endpoint("", "mds0=127.0.0.1:2")  # no raise


if __name__ == "__main__":
    unittest.main()
