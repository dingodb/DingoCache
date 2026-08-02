#!/usr/bin/env bash
# Integration smoke for dfkv_server + dfkv_smoke + dfkvctl. arg1 = build dir.
set -e
BUILD="${1:?build dir}"

# --version: each binary prints its name + a version string and exits 0 (must NOT
# fall through to running the daemon).
for b in dfkv_server dfkv_mds dfkvctl dfkv_bench dfkv_smoke; do
  out=$("$BUILD/$b" --version)
  echo "$out" | grep -qE "^$b [0-9]+\.[0-9]+" || { echo "$b --version bad: '$out'"; exit 1; }
done
echo "version smoke OK"

D=$(mktemp -d)
mkdir "$D/cache"
: > "$D/srv.out"   # pre-create so the awk below can't fail on a missing file under `set -e`
"$BUILD/dfkv_server" --dir "$D/cache" --port 0 --cap 1073741824 >>"$D/srv.out" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null; rm -rf "$D"' EXIT
P=""
# `|| true`: a transient awk read must never trip `set -e` before the server prints PORT.
for i in $(seq 1 100); do P=$(awk '/PORT/{print $2}' "$D/srv.out" 2>/dev/null || true); [ -n "$P" ] && break; sleep 0.05; done
[ -n "$P" ] || { echo "server did not report PORT"; cat "$D/srv.out"; exit 1; }
"$BUILD/dfkv_smoke" --members "n=127.0.0.1:$P" --size 4096
"$BUILD/dfkvctl" --members "n=127.0.0.1:$P" put k v12345
got=$("$BUILD/dfkvctl" --members "n=127.0.0.1:$P" get k)
[ "$got" = "v12345" ] || { echo "get mismatch: '$got'"; exit 1; }
"$BUILD/dfkvctl" --members "n=127.0.0.1:$P" exist k | grep -q true
"$BUILD/dfkvctl" stat "127.0.0.1:$P" | grep -q dfkv_cache_put_total
echo "tool_smoke OK (port $P)"

# Strict arg parsing: an unknown flag / bad number / bad --advertise must fail
# fast (exit 2), not run with silent defaults. (`set -e` is on, so guard the
# expected-nonzero calls.)
rc=0; "$BUILD/dfkv_server" --dir "$D" --capp 5 --port 0 >/dev/null 2>&1 || rc=$?
[ "$rc" = 2 ] || { echo "dfkv_server unknown flag: expected exit 2, got $rc"; exit 1; }
rc=0; "$BUILD/dfkv_server" --dir "$D" --cap 5TiB --port 0 >/dev/null 2>&1 || rc=$?
[ "$rc" = 2 ] || { echo "dfkv_server bad --cap: expected exit 2, got $rc"; exit 1; }
rc=0; "$BUILD/dfkv_server" --dir "$D" --advertise no-colon --port 0 >/dev/null 2>&1 || rc=$?
[ "$rc" = 2 ] || { echo "dfkv_server bad --advertise: expected exit 2, got $rc"; exit 1; }
echo "strict-args smoke OK"

# MDS must fail loud (exit 1) when etcd is unreachable, within the probe window,
# instead of running "healthy" while every registration silently fails.
rc=0; DFKV_MDS_ETCD_PROBE_MS=500 "$BUILD/dfkv_mds" --listen 0 --etcd 127.0.0.1:9 \
  >/dev/null 2>&1 || rc=$?
[ "$rc" = 1 ] || { echo "dfkv_mds bad etcd: expected exit 1, got $rc"; exit 1; }
echo "mds etcd-probe smoke OK"
