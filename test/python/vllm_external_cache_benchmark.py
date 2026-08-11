#!/usr/bin/env python3
"""Deterministic streaming benchmark for vLLM external KV-cache rounds.

Rounds are labels only: this program never clears, resets, or otherwise manages
vLLM's local prefix cache or the configured external cache. Reuse the generated
corpus (automatically from the same output directory, or via --corpus-file) to
compare populate and post-restart hot rounds with byte-identical prompts.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import math
import os
import random
import statistics
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


CORPUS_FORMAT_VERSION = 2
GENERATOR_VERSION = "dfkv-vllm-long-prefix-v1"
DEPLOYMENT_IDENTITY_FIELDS = (
    "model",
    "model_revision",
    "vllm_version",
    "vllm_commit",
    "dfkv_version",
    "dfkv_build_commit",
    "connector_layout",
    "connector_version",
    "deployment_manifest_hash",
)
WORDS = (
    "amber", "aperture", "archive", "balance", "beacon", "binary", "canvas",
    "cascade", "circuit", "context", "delta", "deterministic", "engine",
    "fabric", "gradient", "harbor", "index", "kernel", "lattice", "matrix",
    "memory", "metric", "model", "network", "observe", "parallel", "prefix",
    "protocol", "quartz", "request", "scheduler", "segment", "signal",
    "storage", "stream", "tensor", "token", "vector", "window", "zenith",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_write(path: str, data: bytes) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".tmp-vllm-cache-bench-", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def completion_url(endpoint: str) -> str:
    parsed = urllib.parse.urlsplit(endpoint)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("--endpoint must be an absolute http(s) URL")
    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        result_path = path
    elif path.endswith("/v1"):
        result_path = path + "/chat/completions"
    elif not path:
        result_path = "/v1/chat/completions"
    else:
        result_path = path + "/v1/chat/completions"
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, result_path, parsed.query, ""))


def tokenizer_url(endpoint: str, operation: str) -> str:
    parsed = urllib.parse.urlsplit(endpoint)
    path = parsed.path.rstrip("/")
    marker = path.find("/v1/")
    if marker >= 0:
        path = path[:marker]
    elif path.endswith("/v1"):
        path = path[:-3]
    elif path.endswith("/chat/completions"):
        path = path[: -len("/chat/completions")]
    result_path = path.rstrip("/") + "/" + operation
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, result_path, parsed.query, ""))


def metrics_url(endpoint: str) -> str:
    parsed = urllib.parse.urlsplit(endpoint)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("metrics endpoints must be absolute http(s) URLs")
    path = parsed.path.rstrip("/")
    if not path:
        path = "/metrics"
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, ""))


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=canonical_json(payload),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read(4096).decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} returned HTTP {error.code}: {detail}") from error
    try:
        decoded = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"POST {url} returned invalid JSON") from error
    if not isinstance(decoded, dict):
        raise RuntimeError(f"POST {url} returned a non-object JSON response")
    return decoded


def token_ids(endpoint: str, model: str, text: str, timeout: float) -> list[int]:
    response = post_json(
        tokenizer_url(endpoint, "tokenize"),
        {"model": model, "prompt": text, "add_special_tokens": False},
        timeout,
    )
    tokens = response.get("tokens")
    if not isinstance(tokens, list) or not all(isinstance(item, int) for item in tokens):
        raise RuntimeError("vLLM /tokenize response did not include integer tokens")
    return tokens


def detokenize(endpoint: str, model: str, tokens: list[int], timeout: float) -> str:
    response = post_json(
        tokenizer_url(endpoint, "detokenize"),
        {"model": model, "tokens": tokens},
        timeout,
    )
    prompt = response.get("prompt")
    if not isinstance(prompt, str):
        raise RuntimeError("vLLM /detokenize response did not include prompt text")
    return prompt


def deterministic_text(seed: int, character_count: int) -> str:
    rng = random.Random(seed)
    parts = ["Synthetic external-cache benchmark corpus.\n"]
    length = len(parts[0])
    sentence_words: list[str] = []
    while length < character_count:
        sentence_words.append(WORDS[rng.randrange(len(WORDS))])
        if len(sentence_words) == 16:
            sentence = " ".join(sentence_words).capitalize() + ".\n"
            parts.append(sentence)
            length += len(sentence)
            sentence_words.clear()
    if sentence_words and length < character_count:
        tail = " ".join(sentence_words) + ".\n"
        parts.append(tail)
    return "".join(parts)[:character_count]


def prefix_for_token_target(
    endpoint: str, model: str, target: int, seed: int, timeout: float
) -> tuple[str, int]:
    character_count = max(1024, target * 5)
    while True:
        candidate = deterministic_text(seed, character_count)
        candidate_tokens = token_ids(endpoint, model, candidate, timeout)
        if len(candidate_tokens) >= target + 32:
            break
        character_count *= 2

    low = max(1, target - 64)
    high = min(len(candidate_tokens), target + 64)
    checked: set[int] = set()
    while low <= high:
        token_count = (low + high) // 2
        checked.add(token_count)
        text = detokenize(endpoint, model, candidate_tokens[:token_count], timeout)
        actual = len(token_ids(endpoint, model, text, timeout))
        if actual == target:
            return text, actual
        if actual < target:
            low = token_count + 1
        else:
            high = token_count - 1

    for token_count in range(max(1, target - 128), min(len(candidate_tokens), target + 128) + 1):
        if token_count in checked:
            continue
        text = detokenize(endpoint, model, candidate_tokens[:token_count], timeout)
        actual = len(token_ids(endpoint, model, text, timeout))
        if actual == target:
            return text, actual
    raise RuntimeError(
        f"could not construct a prefix that re-tokenizes to exactly {target} tokens; "
        "use --prefix-chars or reuse a previously saved corpus"
    )


def deployment_identity(args: argparse.Namespace) -> dict[str, str | None]:
    """Return the fixed-schema, operator-supplied deployment identity."""
    return {
        "model": args.model,
        "model_revision": args.model_revision,
        "vllm_version": args.vllm_version,
        "vllm_commit": args.vllm_commit,
        "dfkv_version": args.dfkv_version,
        "dfkv_build_commit": args.dfkv_build_commit,
        "connector_layout": args.connector_layout,
        "connector_version": args.connector_version,
        "deployment_manifest_hash": args.deployment_manifest_hash,
    }


def corpus_compatibility_identity(
    identity: dict[str, str | None],
) -> dict[str, str | None]:
    """Identity fields that affect the corpus model/tokenizer contract."""
    return {
        "model": identity["model"],
        "model_revision": identity["model_revision"],
    }


def require_summary_identity(
    path: str | None, expected: dict[str, str | None]
) -> str | None:
    if path is None:
        return None
    with open(path, "rb") as source:
        raw = source.read()
    summary = json.loads(raw)
    if not isinstance(summary, dict):
        raise ValueError("identity reference summary must be a JSON object")
    actual = summary.get("deployment_identity")
    if not isinstance(actual, dict):
        raise ValueError("identity reference summary is missing deployment_identity")
    if tuple(sorted(actual)) != tuple(sorted(DEPLOYMENT_IDENTITY_FIELDS)):
        raise ValueError("identity reference summary has an invalid deployment_identity schema")
    mismatches = [
        field for field in DEPLOYMENT_IDENTITY_FIELDS
        if actual.get(field) != expected[field]
    ]
    if mismatches:
        raise ValueError(
            "deployment identity does not match reference summary fields: "
            + ", ".join(mismatches)
        )
    return sha256_bytes(raw)


def expected_corpus_config(args: argparse.Namespace) -> dict[str, Any]:
    target_mode = "tokens" if args.prefix_tokens is not None else "characters"
    target_value = args.prefix_tokens if args.prefix_tokens is not None else args.prefix_chars
    return {
        "generator_version": GENERATOR_VERSION,
        "model": args.model,
        "prefix_target_mode": target_mode,
        "prefix_target": target_value,
        "request_count": args.request_count,
        "seed": args.seed,
    }


def validate_corpus(
    corpus: dict[str, Any],
    expected: dict[str, Any],
    current_identity: dict[str, str | None],
) -> None:
    if corpus.get("format_version") != CORPUS_FORMAT_VERSION:
        raise ValueError("unsupported corpus format_version")
    if corpus.get("config") != expected:
        raise ValueError("saved corpus config does not match this command")
    saved_identity = corpus.get("deployment_identity")
    if not isinstance(saved_identity, dict):
        raise ValueError("saved corpus is missing deployment_identity")
    if tuple(sorted(saved_identity)) != tuple(sorted(DEPLOYMENT_IDENTITY_FIELDS)):
        raise ValueError("saved corpus has an invalid deployment_identity schema")
    compatibility = corpus.get("compatibility_identity")
    if compatibility != corpus_compatibility_identity(saved_identity):
        raise ValueError("saved corpus has inconsistent compatibility identity")
    if compatibility != corpus_compatibility_identity(current_identity):
        raise ValueError("saved corpus model identity does not match this command")
    prefix = corpus.get("prefix")
    suffixes = corpus.get("suffixes")
    if not isinstance(prefix, str) or not isinstance(suffixes, list):
        raise ValueError("saved corpus is missing prefix or suffixes")
    if len(suffixes) != expected["request_count"] or not all(isinstance(item, str) for item in suffixes):
        raise ValueError("saved corpus suffix count is invalid")
    content = {key: value for key, value in corpus.items() if key != "corpus_sha256"}
    if corpus.get("corpus_sha256") != sha256_bytes(canonical_json(content)):
        raise ValueError("saved corpus checksum mismatch")


def load_or_create_corpus(args: argparse.Namespace) -> tuple[dict[str, Any], str, bool]:
    expected = expected_corpus_config(args)
    identity = deployment_identity(args)
    config_id = sha256_bytes(canonical_json({
        "config": expected,
        "compatibility_identity": corpus_compatibility_identity(identity),
    }))[:16]
    path = args.corpus_file or os.path.join(args.output_dir, f"corpus-{config_id}.json")
    if os.path.exists(path):
        with open(path, "rb") as source:
            corpus = json.load(source)
        if not isinstance(corpus, dict):
            raise ValueError("saved corpus must be a JSON object")
        validate_corpus(corpus, expected, identity)
        return corpus, os.path.abspath(path), False

    if args.corpus_file:
        raise FileNotFoundError(f"--corpus-file does not exist: {path}")
    if args.prefix_tokens is not None:
        prefix, actual_tokens = prefix_for_token_target(
            args.endpoint, args.model, args.prefix_tokens, args.seed, args.timeout
        )
    else:
        prefix = deterministic_text(args.seed, args.prefix_chars)
        actual_tokens = None

    suffix_rng = random.Random(args.seed ^ 0xD1F5CAFE)
    suffixes = [
        "\n\n[benchmark request %06d nonce %016x]\n"
        "Continue the synthetic document with a concise deterministic analysis."
        % (index, suffix_rng.getrandbits(64))
        for index in range(args.request_count)
    ]
    corpus = {
        "format_version": CORPUS_FORMAT_VERSION,
        "config": expected,
        "deployment_identity": identity,
        "compatibility_identity": corpus_compatibility_identity(identity),
        "prefix": prefix,
        "suffixes": suffixes,
        "prefix_characters": len(prefix),
        "prefix_tokens": actual_tokens,
        "prefix_sha256": sha256_bytes(prefix.encode("utf-8")),
    }
    corpus["corpus_sha256"] = sha256_bytes(canonical_json(corpus))
    atomic_write(path, canonical_json(corpus) + b"\n")
    return corpus, os.path.abspath(path), True


def snapshot_metrics(
    endpoints: list[tuple[str, str]], phase: str, run_stem: str, output_dir: str, timeout: float
) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    for index, (kind, endpoint) in enumerate(endpoints):
        started = utc_now()
        url = metrics_url(endpoint)
        record: dict[str, Any] = {
            "kind": kind,
            "endpoint_sha256": sha256_bytes(endpoint.encode("utf-8")),
            "phase": phase,
            "timestamp": started,
        }
        try:
            request = urllib.request.Request(url, headers={"Accept": "text/plain"}, method="GET")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
                status = response.status
            filename = f"{run_stem}.metrics-{phase}-{kind}-{index}.prom"
            path = os.path.join(output_dir, filename)
            atomic_write(path, body)
            record.update({
                "success": True,
                "http_status": status,
                "bytes": len(body),
                "sha256": sha256_bytes(body),
                "artifact": filename,
            })
        except (OSError, urllib.error.URLError, ValueError) as error:
            record.update({"success": False, "error_type": type(error).__name__})
        snapshots.append(record)
    return snapshots


def stream_request(
    args: argparse.Namespace,
    corpus: dict[str, Any],
    request_index: int,
    run_started_ns: int,
    start_gate: threading.Event,
    active_state: dict[str, Any],
) -> dict[str, Any]:
    start_gate.wait()
    prompt = corpus["prefix"] + corpus["suffixes"][request_index]
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "seed": args.seed + request_index,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started_at = utc_now()
    started_ns = time.perf_counter_ns()
    with active_state["lock"]:
        active_state["active"] += 1
        active_state["maximum"] = max(active_state["maximum"], active_state["active"])
        active_at_start = active_state["active"]

    first_token_ns: int | None = None
    completion_hasher = hashlib.sha256()
    completion_characters = 0
    output_chunks = 0
    usage_completion_tokens: int | None = None
    finish_reason: str | None = None
    saw_done = False
    error_text: str | None = None
    http_status: int | None = None
    try:
        request = urllib.request.Request(
            completion_url(args.endpoint),
            data=canonical_json(payload),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            http_status = response.status
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="strict").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    saw_done = True
                    break
                if not data:
                    continue
                event = json.loads(data)
                if not isinstance(event, dict):
                    continue
                if "error" in event:
                    raise RuntimeError("stream reported an error")
                usage = event.get("usage")
                if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
                    usage_completion_tokens = usage["completion_tokens"]
                choices = event.get("choices")
                if not isinstance(choices, list):
                    continue
                for choice in choices:
                    if not isinstance(choice, dict):
                        continue
                    if choice.get("finish_reason") is not None:
                        finish_reason = str(choice["finish_reason"])
                    delta = choice.get("delta")
                    if not isinstance(delta, dict):
                        continue
                    pieces = []
                    for field in ("reasoning_content", "content"):
                        value = delta.get(field)
                        if isinstance(value, str) and value:
                            pieces.append(value)
                    for piece in pieces:
                        if first_token_ns is None:
                            first_token_ns = time.perf_counter_ns()
                        encoded = piece.encode("utf-8")
                        completion_hasher.update(encoded)
                        completion_characters += len(piece)
                        output_chunks += 1
    except urllib.error.HTTPError as error:
        error.read(4096)
        error_text = f"HTTPError: HTTP {error.code}"
        http_status = error.code
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RuntimeError) as error:
        error_text = type(error).__name__
    finally:
        ended_ns = time.perf_counter_ns()
        ended_at = utc_now()
        with active_state["lock"]:
            active_state["active"] -= 1

    success = error_text is None and first_token_ns is not None and (saw_done or finish_reason is not None)
    output_tokens = usage_completion_tokens if usage_completion_tokens is not None else output_chunks
    return {
        "request_index": request_index,
        "round_name": args.round_name,
        "success": success,
        "error": error_text,
        "http_status": http_status,
        "started_at": started_at,
        "ended_at": ended_at,
        "scheduled_offset_seconds": (started_ns - run_started_ns) / 1e9,
        "ttft_seconds": (first_token_ns - started_ns) / 1e9 if first_token_ns is not None else None,
        "e2e_seconds": (ended_ns - started_ns) / 1e9,
        "output_tokens": output_tokens,
        "output_tokens_source": "usage.completion_tokens" if usage_completion_tokens is not None else "nonempty_stream_chunks",
        "output_chunks": output_chunks,
        "output_characters": completion_characters,
        "output_sha256": completion_hasher.hexdigest(),
        "finish_reason": finish_reason,
        "saw_done": saw_done,
        "configured_concurrency": args.concurrency,
        "active_at_start": active_at_start,
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "corpus_sha256": corpus["corpus_sha256"],
    }


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def latency_summary(values: list[float]) -> dict[str, float | None]:
    return {
        "min": min(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, help="vLLM base URL or /v1/chat/completions URL")
    parser.add_argument("--model", required=True)
    identity = parser.add_argument_group("deployment identity")
    identity.add_argument("--model-revision", help="exact model revision or artifact digest")
    identity.add_argument("--vllm-version", help="exact deployed vLLM version")
    identity.add_argument("--vllm-commit", help="exact deployed vLLM source commit")
    identity.add_argument("--dfkv-version", help="exact deployed dfkv version")
    identity.add_argument("--dfkv-build-commit", help="exact deployed libdfkv build commit")
    identity.add_argument("--connector-layout", help="exact connector storage layout identity")
    identity.add_argument("--connector-version", help="exact deployed connector package version")
    identity.add_argument(
        "--deployment-manifest-hash",
        help="exact digest of the immutable deployment manifest",
    )
    identity.add_argument(
        "--require-identity-summary",
        metavar="FILE",
        help="fail before requests unless FILE has the same deployment identity",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--prefix-tokens", type=int, help="exact prefix token target via vLLM tokenize APIs")
    target.add_argument("--prefix-chars", type=int, help="exact prefix character target")
    parser.add_argument("--request-count", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--round-name", required=True, choices=("cold", "populate", "hot"),
        help="measurement label only; never changes or resets cache state",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--corpus-file",
        help="reuse a saved corpus; generation config and model identity must match",
    )
    parser.add_argument(
        "--vllm-metrics-endpoint", action="append", default=[],
        help="optional vLLM Prometheus URL; repeat for multiple endpoints",
    )
    parser.add_argument(
        "--dfkv-metrics-endpoint", action="append", default=[],
        help="optional dfkv Prometheus URL; repeat for multiple endpoints",
    )
    parser.add_argument("--timeout", type=float, default=600.0, help="HTTP timeout in seconds")
    args = parser.parse_args(argv)
    for name in ("request_count", "concurrency", "max_tokens"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    target_value = args.prefix_tokens if args.prefix_tokens is not None else args.prefix_chars
    if target_value is None or target_value <= 0:
        parser.error("prefix target must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    completion_url(args.endpoint)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    current_identity = deployment_identity(args)
    identity_reference_sha256 = require_summary_identity(
        args.require_identity_summary, current_identity
    )
    os.makedirs(args.output_dir, exist_ok=True)

    corpus, corpus_path, corpus_created = load_or_create_corpus(args)
    timestamp_stem = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_stem = f"{args.round_name}-{timestamp_stem}"
    raw_path = os.path.abspath(os.path.join(args.output_dir, run_stem + ".requests.jsonl"))
    summary_path = os.path.abspath(os.path.join(args.output_dir, run_stem + ".summary.json"))
    metric_endpoints = (
        [("vllm", item) for item in args.vllm_metrics_endpoint]
        + [("dfkv", item) for item in args.dfkv_metrics_endpoint]
    )

    started_at = utc_now()
    before_snapshots = snapshot_metrics(
        metric_endpoints, "before", run_stem, args.output_dir, args.timeout
    )
    active_state: dict[str, Any] = {"active": 0, "maximum": 0, "lock": threading.Lock()}
    start_gate = threading.Event()
    run_started_ns = time.perf_counter_ns()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                stream_request, args, corpus, index, run_started_ns, start_gate, active_state
            )
            for index in range(args.request_count)
        ]
        start_gate.set()
        rows = [future.result() for future in concurrent.futures.as_completed(futures)]
    run_ended_ns = time.perf_counter_ns()
    ended_at = utc_now()
    after_snapshots = snapshot_metrics(
        metric_endpoints, "after", run_stem, args.output_dir, args.timeout
    )

    rows.sort(key=lambda row: row["request_index"])
    raw = b"".join(canonical_json(row) + b"\n" for row in rows)
    atomic_write(raw_path, raw)
    successful = [row for row in rows if row["success"]]
    ttft = [float(row["ttft_seconds"]) for row in successful]
    e2e = [float(row["e2e_seconds"]) for row in successful]
    source_counts: dict[str, int] = {}
    for row in rows:
        source = str(row["output_tokens_source"])
        source_counts[source] = source_counts.get(source, 0) + 1

    config = {
        "endpoint_sha256": sha256_bytes(args.endpoint.encode("utf-8")),
        "model": args.model,
        "prefix_tokens": args.prefix_tokens,
        "prefix_chars": args.prefix_chars,
        "request_count": args.request_count,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "round_name": args.round_name,
        "corpus_file": os.path.basename(corpus_path),
        "vllm_metrics_endpoint_count": len(args.vllm_metrics_endpoint),
        "dfkv_metrics_endpoint_count": len(args.dfkv_metrics_endpoint),
        "timeout": args.timeout,
    }
    summary = {
        "format_version": 2,
        "deployment_identity": current_identity,
        "identity_requirement": {
            "enforced": identity_reference_sha256 is not None,
            "reference_summary_sha256": identity_reference_sha256,
        },
        "config": config,
        "timestamps": {"started_at": started_at, "ended_at": ended_at},
        "duration_seconds": (run_ended_ns - run_started_ns) / 1e9,
        "cache_policy": "round name is a label; benchmark never resets external or local cache",
        "corpus": {
            "artifact": os.path.basename(corpus_path),
            "created_by_this_run": corpus_created,
            "sha256": corpus["corpus_sha256"],
            "prefix_sha256": corpus["prefix_sha256"],
            "prefix_characters": corpus["prefix_characters"],
            "prefix_tokens": corpus["prefix_tokens"],
            "deployment_identity": corpus["deployment_identity"],
            "compatibility_identity": corpus["compatibility_identity"],
        },
        "requests": {
            "raw_jsonl_artifact": os.path.basename(raw_path),
            "total": len(rows),
            "successful": len(successful),
            "failed": len(rows) - len(successful),
            "success_rate": len(successful) / len(rows),
        },
        "concurrency": {
            "configured": args.concurrency,
            "effective_limit": min(args.concurrency, args.request_count),
            "maximum_observed": active_state["maximum"],
        },
        "ttft_seconds": latency_summary(ttft),
        "e2e_seconds": latency_summary(e2e),
        "output_tokens": {
            "total": sum(int(row["output_tokens"]) for row in successful),
            "mean_per_success": (
                statistics.fmean(int(row["output_tokens"]) for row in successful)
                if successful else None
            ),
            "source_request_counts": source_counts,
        },
        "metrics_snapshots": {"before": before_snapshots, "after": after_snapshots},
    }
    atomic_write(summary_path, json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n")
    print(json.dumps({
        "summary": summary_path,
        "raw_jsonl": raw_path,
        "corpus": corpus_path,
        "successful": len(successful),
        "failed": len(rows) - len(successful),
    }, sort_keys=True))
    return 0 if len(successful) == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
