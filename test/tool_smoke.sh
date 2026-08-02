#!/usr/bin/env bash
# Integration smoke for dfkv_server + dfkv_smoke + dfkvctl + dfkv_bench. arg1 = build dir.
set -e
BUILD="${1:?build dir}"

# --version: each binary prints its name + a version string and exits 0 (must NOT
# fall through to running the daemon).
for b in dfkv_server dfkv_mds dfkvctl dfkv_bench dfkv_smoke; do
  out=$("$BUILD/$b" --version)
  echo "$out" | grep -qE "^$b [0-9]+\.[0-9]+" || { echo "$b --version bad: '$out'"; exit 1; }
done
echo "version smoke OK"
"$BUILD/dfkv_server" --help | grep -q \
  'storage backend: slab | file (default slab)'

D=$(mktemp -d)
mkdir "$D/cache"
: > "$D/srv.out"   # pre-create so the awk below can't fail on a missing file under `set -e`
env -u DFKV_STORE_ENGINE -u DFKV_SLAB_WRITE "$BUILD/dfkv_server" \
  --dir "$D/cache" --port 0 --cap 1073741824 >>"$D/srv.out" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null; rm -rf "$D"' EXIT
P=""
# `|| true`: a transient awk read must never trip `set -e` before the server prints PORT.
for i in $(seq 1 100); do P=$(awk '/PORT/{print $2}' "$D/srv.out" 2>/dev/null || true); [ -n "$P" ] && break; sleep 0.05; done
[ -n "$P" ] || { echo "server did not report PORT"; cat "$D/srv.out"; exit 1; }
[ -f "$D/cache/slots.tbl" ] || {
  echo "default server did not create slab metadata"
  cat "$D/srv.out"
  exit 1
}
[ ! -d "$D/cache/blocks" ] || {
  echo "default server silently selected the file engine"
  exit 1
}
"$BUILD/dfkv_smoke" --members "n=127.0.0.1:$P" --size 4096
"$BUILD/dfkvctl" --members "n=127.0.0.1:$P" --namespace "dfkv/smoke" put k v12345
got=$("$BUILD/dfkvctl" --members "n=127.0.0.1:$P" --namespace "dfkv/smoke" get k)
[ "$got" = "v12345" ] || { echo "get mismatch: '$got'"; exit 1; }
"$BUILD/dfkvctl" --members "n=127.0.0.1:$P" --namespace "dfkv/smoke" exist k | grep -q true
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
rc=0; DFKV_MDS_REGISTRATION_TIMEOUT_MS=garbage "$BUILD/dfkv_server" \
  --dir "$D" --port 0 --cap 5 >/dev/null 2>&1 || rc=$?
[ "$rc" = 2 ] || {
  echo "dfkv_server bad MDS registration timeout env: expected exit 2, got $rc"
  exit 1
}
rc=0; "$BUILD/dfkv_server" --dir "$D" --port 0 --cap 5 \
  --mds-registration-timeout-ms 600001 >/dev/null 2>&1 || rc=$?
[ "$rc" = 2 ] || {
  echo "dfkv_server out-of-range MDS registration timeout: expected exit 2, got $rc"
  exit 1
}
echo "strict-args smoke OK"
# Invalid slab geometry must refuse startup, and the explicit flag must beat a
# conflicting file environment. A hidden fallback would run until timeout 124.
mkdir "$D/invalid-slab"
rc=0
DFKV_STORE_ENGINE=file timeout 3s "$BUILD/dfkv_server" \
  --dir "$D/invalid-slab" --port 0 --cap 1073741825 \
  --store-engine slab >/dev/null 2>&1 || rc=$?
[ "$rc" = 1 ] || {
  echo "invalid default slab geometry: expected exit 1, got $rc"
  exit 1
}
[ ! -d "$D/invalid-slab/blocks" ] || {
  echo "invalid slab geometry silently fell back to file"
  exit 1
}
echo "slab-default fail-closed smoke OK"

# A cache node configured for MDS must exit fail-closed instead of remaining an
# active, permanently-unready data listener when its first registration cannot
# succeed. exit 124 would mean GNU timeout killed a hung daemon.
mkdir "$D/unregistered"
rc=0
DFKV_STORE_ENGINE=file timeout 5s "$BUILD/dfkv_server" \
  --dir "$D/unregistered" --port 0 --cap 1048576 \
  --mds 127.0.0.1:1 --group deadline-smoke --id deadline-node \
  --advertise 127.0.0.1:28000 --mds-registration-timeout-ms 1000 \
  >"$D/unregistered.out" 2>&1 || rc=$?
[ "$rc" = 1 ] || {
  echo "dfkv_server first-registration deadline: expected exit 1, got $rc"
  exit 1
}
grep -qE 'DFKV_STORE_ENGINE[[:space:]]*= file[[:space:]]+[(]env[)]' \
  "$D/unregistered.out" || {
  echo "DFKV_STORE_ENGINE=file did not select the diagnostic file engine"
  exit 1
}
[ ! -f "$D/unregistered/slots.tbl" ] || {
  echo "explicit file engine unexpectedly created slab metadata"
  exit 1
}
echo "MDS first-registration deadline smoke OK"

# dfkv_bench has automation-facing exit semantics: malformed CLI is 2, a
# completed phase with failed operations is 1, and only zero-failure phases are 0.
expect_bench_usage_error() {
  label=$1
  shift
  rc=0
  "$BUILD/dfkv_bench" "$@" >/dev/null 2>&1 || rc=$?
  [ "$rc" = 2 ] || {
    echo "dfkv_bench $label: expected exit 2, got $rc"
    exit 1
  }
}
expect_bench_usage_error "unknown flag" --members "n=127.0.0.1:$P" --bogus x
expect_bench_usage_error "dangling flag" --members "n=127.0.0.1:$P" --size
expect_bench_usage_error "bad number" --members "n=127.0.0.1:$P" --count nope
expect_bench_usage_error "overflowing number" --members "n=127.0.0.1:$P" \
  --count 184467440737095516160
expect_bench_usage_error "invalid op" --members "n=127.0.0.1:$P" --op delete
expect_bench_usage_error "zero size" --members "n=127.0.0.1:$P" --size 0
expect_bench_usage_error "zero count" --members "n=127.0.0.1:$P" --count 0
expect_bench_usage_error "zero threads" --members "n=127.0.0.1:$P" --threads 0
expect_bench_usage_error "zero batch" --members "n=127.0.0.1:$P" --batch 0

out=$("$BUILD/dfkv_bench" --members "n=127.0.0.1:$P" --size 4096 --count 4 \
  --threads 2 --batch 2 --op both --key-seed "tool-smoke-$$")
echo "$out" | grep -qE '^PUT[[:space:]].*fails=0$' || {
  echo "dfkv_bench successful PUT report missing: '$out'"
  exit 1
}
echo "$out" | grep -qE '^GET[[:space:]].*fails=0$' || {
  echo "dfkv_bench successful GET report missing: '$out'"
  exit 1
}

rc=0
out=$("$BUILD/dfkv_bench" --members "n=127.0.0.1:$P" --size 4096 --count 2 \
  --threads 1 --batch 1 --op get --key-seed "tool-smoke-missing-$$") || rc=$?
[ "$rc" = 1 ] || { echo "dfkv_bench misses: expected exit 1, got $rc"; exit 1; }
echo "$out" | grep -qE '^GET[[:space:]].*fails=2$' || {
  echo "dfkv_bench failed GET report missing: '$out'"
  exit 1
}
echo "dfkv_bench parser/exit smoke OK"

# MDS must fail loud (exit 1) when etcd is unreachable, within the probe window,
# instead of running "healthy" while every registration silently fails.
rc=0; DFKV_MDS_ETCD_PROBE_MS=500 "$BUILD/dfkv_mds" --listen 0 --etcd 127.0.0.1:9 \
  >/dev/null 2>&1 || rc=$?
[ "$rc" = 1 ] || { echo "dfkv_mds bad etcd: expected exit 1, got $rc"; exit 1; }
echo "mds etcd-probe smoke OK"
