#!/usr/bin/env bash
# Single source of truth for shared connector modules lives alongside the SGLang
# HiCache plugin (which imports them unvendored): the dfkv_telemetry package and
# the dfkv_hot_config watcher. The vLLM and LMCache connectors are independent
# pip wheels that can't import a repo-level module at runtime, so they vendor a
# byte-identical copy (dfkv_telemetry -> _telemetry/, dfkv_hot_config.py ->
# _hot_config.py). Run this after editing the canonical files; CI guards against
# drift via test/python/test_telemetry_vendor_sync.py.
set -euo pipefail
cd "$(dirname "$0")/.."

SRC="integration/hicache/dfkv_telemetry"
FILES="__init__.py config.py metrics_push.py otlp_json.py tracing.py otlp_traces.py"
DSTS=(
  "integration/vllm/src/dfkv_vllm/_telemetry"
  "integration/lmcache/src/dfkv_connector/_telemetry"
)

for dst in "${DSTS[@]}"; do
  mkdir -p "$dst"
  for f in $FILES; do
    cp "$SRC/$f" "$dst/$f"
  done
  echo "synced $SRC -> $dst"
done

# Standalone hot-config watcher module -> _hot_config.py in each package root.
HOT_SRC="integration/hicache/dfkv_hot_config.py"
HOT_DSTS=(
  "integration/vllm/src/dfkv_vllm/_hot_config.py"
  "integration/lmcache/src/dfkv_connector/_hot_config.py"
)
for dst in "${HOT_DSTS[@]}"; do
  cp "$HOT_SRC" "$dst"
  echo "synced $HOT_SRC -> $dst"
done

# Connector access log: the vLLM copy is canonical (HiCache has its own).
ACCESS_SRC="integration/vllm/src/dfkv_vllm/access_log.py"
ACCESS_DST="integration/lmcache/src/dfkv_connector/access_log.py"
cp "$ACCESS_SRC" "$ACCESS_DST"
echo "synced $ACCESS_SRC -> $ACCESS_DST"
