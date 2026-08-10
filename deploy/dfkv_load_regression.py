#!/usr/bin/env python3
"""Repeatable dfkv candidate-vs-baseline load regression gate."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

from dfkv_ops_common import (
    atomic_write_with_checksum,
    parse_bench,
    parse_bench_diagnostics,
)


_METRIC = re.compile(r'^dfkv_op_latency_seconds_bucket\{([^}]*)\}\s+([0-9.eE+-]+)$')
_LABEL = re.compile(r'(\w+)="((?:\\.|[^"\\])*)"')


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def scrape_histograms(endpoints: str, timeout: float) -> dict[str, dict[float, float]]:
    totals: dict[str, dict[float, float]] = {"put": {}, "get": {}}
    for raw_endpoint in [item for item in endpoints.split(",") if item]:
        url = raw_endpoint if "://" in raw_endpoint else "http://" + raw_endpoint
        if not url.rstrip("/").endswith("/metrics"):
            url = url.rstrip("/") + "/metrics"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                text = response.read().decode()
        except (OSError, urllib.error.URLError, UnicodeDecodeError) as error:
            raise RuntimeError(f"metrics scrape failed for {url}: {error}") from error
        seen: set[tuple[str, float]] = set()
        for line in text.splitlines():
            match = _METRIC.match(line)
            if not match:
                continue
            labels = {item.group(1): item.group(2) for item in _LABEL.finditer(match.group(1))}
            op = labels.get("op")
            if op not in totals or "le" not in labels:
                continue
            bound = math.inf if labels["le"] == "+Inf" else float(labels["le"])
            key = (op, bound)
            if key in seen:
                raise ValueError(f"duplicate histogram series for op={op} le={labels['le']} at {url}")
            seen.add(key)
            totals[op][bound] = totals[op].get(bound, 0.0) + float(match.group(2))
    for op in ("put", "get"):
        if math.inf not in totals[op] or len(totals[op]) < 2:
            raise ValueError(f"metrics endpoints did not expose dfkv_op_latency_seconds_bucket for op={op}")
    return totals


def histogram_delta(
    before: dict[str, dict[float, float]], after: dict[str, dict[float, float]]
) -> dict[str, dict[float, float]]:
    result: dict[str, dict[float, float]] = {}
    for op in ("put", "get"):
        if set(before[op]) != set(after[op]):
            raise ValueError(f"histogram bucket layout changed during {op} benchmark")
        result[op] = {}
        for bound in before[op]:
            delta = after[op][bound] - before[op][bound]
            if delta < 0:
                raise ValueError(f"histogram reset during {op} benchmark")
            result[op][bound] = delta
        if result[op][math.inf] <= 0:
            raise ValueError(f"no sampled {op} latency observations during measured workload")
    return result


def histogram_quantile(buckets: dict[float, float], quantile: float) -> float:
    total = buckets[math.inf]
    rank = total * quantile
    previous_bound = 0.0
    previous_count = 0.0
    for bound in sorted(buckets):
        count = buckets[bound]
        if count >= rank:
            if math.isinf(bound):
                raise ValueError(f"p{int(quantile * 100)} latency falls in +Inf bucket")
            bucket_count = count - previous_count
            if bucket_count <= 0:
                return bound
            fraction = max(0.0, min(1.0, (rank - previous_count) / bucket_count))
            return previous_bound + (bound - previous_bound) * fraction
        previous_bound, previous_count = bound, count
    raise ValueError("malformed cumulative histogram")


def target_args(args: argparse.Namespace, name: str) -> list[str]:
    members = getattr(args, f"{name}_members")
    mds = getattr(args, f"{name}_mds")
    result = ["--members", members] if members else ["--mds", mds, "--group", getattr(args, f"{name}_group")]
    return result


def bench_once(
    args: argparse.Namespace,
    name: str,
    count: int,
    seed: str,
    *,
    allow_operation_failures: bool,
) -> dict[str, dict[str, float | int]]:
    binary = getattr(args, f"{name}_bench") or args.dfkv_bench
    command = [
        binary,
        *target_args(args, name),
        "--size", str(args.size),
        "--count", str(count),
        "--threads", str(args.threads),
        "--batch", str(args.batch),
        "--bc", str(args.bc),
        "--op", "both",
        "--key-seed", seed,
        "--ready-timeout", str(int(args.ready_timeout)),
    ]
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=args.run_timeout,
        env=os.environ.copy(),
        check=False,
    )
    legacy_ready_timeout = False
    option_error = (completed.stderr + "\n" + completed.stdout).lower()
    if (completed.returncode == 2
            and "--ready-timeout" in option_error
            and ("unknown" in option_error or "unrecognized" in option_error
                 or "usage:" in option_error)):
        option_index = command.index("--ready-timeout")
        del command[option_index:option_index + 2]
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.run_timeout,
            env=os.environ.copy(),
            check=False,
        )
        legacy_ready_timeout = True
    diagnostics = parse_bench_diagnostics(completed.stdout, required=False)
    if completed.returncode not in (0, 1):
        detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic"
        if diagnostics:
            detail += "; diagnostics=" + json.dumps(
                diagnostics, sort_keys=True, separators=(",", ":"))
        raise RuntimeError(f"dfkv_bench exited {completed.returncode}: {detail}")
    phases = parse_bench(completed.stdout)
    for phase, values in phases.items():
        values["diagnostics_available"] = phase in diagnostics
        if phase in diagnostics:
            values["diagnostics"] = diagnostics[phase]
    if legacy_ready_timeout:
        for phase in phases.values():
            phase["compatibility"] = {
                "ready_timeout_supported": False,
                "diagnostics_available": bool(phase["diagnostics_available"]),
                "readiness": "legacy dfkv_bench behavior",
            }
    failed = sum(int(phase["fails"]) for phase in phases.values())
    # Current dfkv_bench exits 1 when any data op fails. Preserve those complete
    # results for the measured error-rate gate; rc=1 without reported failures
    # remains an execution error, and warmup failures always fail closed.
    if completed.returncode == 1 and failed == 0:
        raise RuntimeError("dfkv_bench exited 1 without reporting operation failures")
    if failed and not allow_operation_failures:
        raise RuntimeError(f"warmup reported {failed} failed operation(s)")
    return phases


def run_target(args: argparse.Namespace, name: str, nonce: str) -> dict[str, object]:
    metrics_endpoint = getattr(args, f"{name}_metrics")
    # Fail before mutating the target workload when server-side histogram
    # evidence is unreachable or incomplete.
    scrape_histograms(metrics_endpoint, args.metrics_timeout)
    for run in range(args.warmup_runs):
        bench_once(
            args, name, args.warmup_count,
            f"loadreg-{nonce}-{name}-warmup-{run}",
            allow_operation_failures=False,
        )
    before = scrape_histograms(metrics_endpoint, args.metrics_timeout)
    trials: list[dict[str, object]] = []
    for run in range(args.runs):
        phases = bench_once(
            args, name, args.count, f"loadreg-{nonce}-{name}-measure-{run}",
            allow_operation_failures=True,
        )
        trials.append({"run": run + 1, "phases": phases})
    after = scrape_histograms(metrics_endpoint, args.metrics_timeout)
    latency_delta = histogram_delta(before, after)

    phase_summaries: dict[str, object] = {}
    for op in ("put", "get"):
        phase_rows = [trial["phases"][op] for trial in trials]  # type: ignore[index]
        throughput = [float(row["throughput_gbps"]) for row in phase_rows]
        total_fails = sum(int(row["fails"]) for row in phase_rows)
        total_ops = sum(int(row["count"]) for row in phase_rows)
        phase_summaries[op] = {
            "throughput_gbps": summarize(throughput),
            "latency_ms": {
                "median": histogram_quantile(latency_delta[op], 0.50) * 1000.0,
                "p95": histogram_quantile(latency_delta[op], 0.95) * 1000.0,
                "p99": histogram_quantile(latency_delta[op], 0.99) * 1000.0,
                "sampled_operations": int(latency_delta[op][math.inf]),
                "source": "delta of dfkv_op_latency_seconds histogram",
            },
            "error_rate": total_fails / total_ops if total_ops else 1.0,
            "failed_operations": total_fails,
            "operations": total_ops,
        }
    return {"trials": trials, "summary": phase_summaries}


def compare(args: argparse.Namespace, baseline: dict[str, object], candidate: dict[str, object]) -> list[dict[str, object]]:
    regressions: list[dict[str, object]] = []
    base_summary = baseline["summary"]
    cand_summary = candidate["summary"]
    assert isinstance(base_summary, dict) and isinstance(cand_summary, dict)
    for op in ("put", "get"):
        base = base_summary[op]
        cand = cand_summary[op]
        assert isinstance(base, dict) and isinstance(cand, dict)
        base_thr = float(base["throughput_gbps"]["median"])  # type: ignore[index]
        cand_thr = float(cand["throughput_gbps"]["median"])  # type: ignore[index]
        throughput_regression = 1.0 - cand_thr / base_thr if base_thr > 0 else math.inf
        if throughput_regression > args.max_throughput_regression / 100.0:
            regressions.append(
                {
                    "phase": op,
                    "metric": "median_throughput_gbps",
                    "baseline": base_thr,
                    "candidate": cand_thr,
                    "regression_percent": throughput_regression * 100.0,
                    "threshold_percent": args.max_throughput_regression,
                }
            )
        for quantile in ("p95", "p99"):
            base_latency = float(base["latency_ms"][quantile])  # type: ignore[index]
            cand_latency = float(cand["latency_ms"][quantile])  # type: ignore[index]
            latency_regression = cand_latency / base_latency - 1.0 if base_latency > 0 else math.inf
            if latency_regression > args.max_latency_regression / 100.0:
                regressions.append(
                    {
                        "phase": op,
                        "metric": f"{quantile}_latency_ms",
                        "baseline": base_latency,
                        "candidate": cand_latency,
                        "regression_percent": latency_regression * 100.0,
                        "threshold_percent": args.max_latency_regression,
                    }
                )
        base_error = float(base["error_rate"])
        candidate_error = float(cand["error_rate"])
        if base_error > args.max_error_rate / 100.0:
            raise RuntimeError(
                f"baseline {op} error rate {base_error * 100:.6f}% exceeds allowed {args.max_error_rate:.6f}%"
            )
        if candidate_error > args.max_error_rate / 100.0:
            regressions.append(
                {
                    "phase": op,
                    "metric": "error_rate",
                    "baseline": base_error,
                    "candidate": candidate_error,
                    "candidate_percent": candidate_error * 100.0,
                    "threshold_percent": args.max_error_rate,
                }
            )
    return regressions


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Run identical warmed dfkv_bench workloads against baseline and candidate, "
            "derive throughput trial median/p95/p99 and latency median/p95/p99 from "
            "server histogram deltas, write a JSON artifact, and exit 3 on regression."
        )
    )
    result.add_argument("--baseline-members", help="baseline static name=host:port list")
    result.add_argument("--baseline-mds", help="baseline MDS endpoint list")
    result.add_argument("--baseline-group", default="default", help="baseline MDS group")
    result.add_argument("--candidate-members", help="candidate static name=host:port list")
    result.add_argument("--candidate-mds", help="candidate MDS endpoint list")
    result.add_argument("--candidate-group", default="default", help="candidate MDS group")
    result.add_argument("--baseline-metrics", required=True, help="comma-separated baseline server metrics URLs/host:ports")
    result.add_argument("--candidate-metrics", required=True, help="comma-separated candidate server metrics URLs/host:ports")
    result.add_argument("--dfkv-bench", default="dfkv_bench", help="default dfkv_bench executable")
    result.add_argument("--baseline-bench", help="optional baseline-specific dfkv_bench executable")
    result.add_argument("--candidate-bench", help="optional candidate-specific dfkv_bench executable")
    result.add_argument("--size", type=int, default=2752512, help="object size bytes")
    result.add_argument("--count", type=int, default=8000, help="operations per measured phase/trial")
    result.add_argument("--threads", type=int, default=64, help="external load threads")
    result.add_argument("--batch", type=int, default=1, help="operations per batch call")
    result.add_argument("--bc", type=int, default=1, help="KVClient internal batch concurrency")
    result.add_argument("--warmup-runs", type=int, default=1, help="unmeasured warmup runs per target")
    result.add_argument("--warmup-count", type=int, default=512, help="operations per warmup phase")
    result.add_argument("--runs", type=int, default=5, help="measured trials per target")
    result.add_argument("--ready-timeout", type=float, default=30.0, help="dfkv_bench MDS readiness timeout seconds")
    result.add_argument("--run-timeout", type=float, default=600.0, help="hard timeout for each bench process seconds")
    result.add_argument("--metrics-timeout", type=float, default=5.0, help="timeout for each metrics scrape seconds")
    result.add_argument("--max-throughput-regression", type=float, default=10.0, help="maximum candidate median throughput loss percent")
    result.add_argument("--max-latency-regression", type=float, default=20.0, help="maximum candidate p95/p99 latency increase percent")
    result.add_argument("--max-error-rate", type=float, default=0.01, help="maximum operation error rate percent")
    result.add_argument("--output", required=True, help="JSON artifact path (written atomically)")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    for name in ("baseline", "candidate"):
        if bool(getattr(args, f"{name}_members")) == bool(getattr(args, f"{name}_mds")):
            parser().error(f"provide exactly one of --{name}-members or --{name}-mds")
    positive = [args.size, args.count, args.threads, args.batch, args.bc, args.warmup_count, args.runs]
    if any(value < 1 for value in positive) or args.warmup_runs < 1:
        parser().error("workload sizes, concurrency, runs, and warmup must be >= 1")
    if args.ready_timeout <= 0 or args.run_timeout <= 0 or args.metrics_timeout <= 0:
        parser().error("timeouts must be positive")
    for value, name in (
        (args.max_throughput_regression, "max throughput regression"),
        (args.max_latency_regression, "max latency regression"),
        (args.max_error_rate, "max error rate"),
    ):
        if value < 0 or value > 100:
            parser().error(f"{name} must be between 0 and 100 percent")

    nonce = f"{int(time.time())}-{uuid.uuid4().hex[:12]}"
    artifact: dict[str, object] = {
        "schema_version": 1,
        "generated_at": int(time.time()),
        "workload": {
            "size_bytes": args.size,
            "count_per_phase": args.count,
            "threads": args.threads,
            "batch": args.batch,
            "batch_concurrency": args.bc,
            "warmup_runs": args.warmup_runs,
            "warmup_count": args.warmup_count,
            "measured_runs": args.runs,
            "transport_environment": {
                key: os.environ[key]
                for key in sorted(os.environ)
                if key.startswith("DFKV_RDMA") or key in {"DFKV_FANOUT_THREADS", "DFKV_BATCH_CONCURRENCY"}
            },
        },
        "thresholds_percent": {
            "max_median_throughput_regression": args.max_throughput_regression,
            "max_p95_p99_latency_regression": args.max_latency_regression,
            "max_error_rate": args.max_error_rate,
        },
    }
    try:
        artifact["baseline"] = run_target(args, "baseline", nonce)
        artifact["candidate"] = run_target(args, "candidate", nonce)
        regressions = compare(args, artifact["baseline"], artifact["candidate"])  # type: ignore[arg-type]
        artifact["regressions"] = regressions
        artifact["status"] = "regression" if regressions else "pass"
        atomic_write_with_checksum(
            args.output, json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        if regressions:
            print(f"REGRESSION: {len(regressions)} threshold violation(s); artifact={args.output}", file=sys.stderr)
            return 3
        print(f"PASS: candidate is within configured thresholds; artifact={args.output}")
        return 0
    except BaseException as error:
        artifact["status"] = "error"
        artifact["error"] = str(error)
        atomic_write_with_checksum(
            args.output, json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        print(f"ERROR: load regression run failed: {error}; artifact={args.output}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
