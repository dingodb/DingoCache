#!/usr/bin/env bash
# Integration smoke for dfkv_server + dfkv_smoke + dfkvctl + dfkv_bench. arg1 = build dir.
set -e
BUILD="${1:?build dir}"

# --version: each binary prints its name + a version string and exits 0 (must NOT
# fall through to running the daemon).
for b in dfkv_server dfkv_mds dfkvctl dfkv_bench dfkv_smoke \
         dfkv_discover_smoke dfkv_gpu_dedup_repro; do
  out=$("$BUILD/$b" --version)
  echo "$out" | grep -qE "^$b [0-9]+\.[0-9]+" || { echo "$b --version bad: '$out'"; exit 1; }
done
echo "version smoke OK"
"$BUILD/dfkv_server" --help | grep -q \
  'storage backend: slab | file (default slab)'

D=$(mktemp -d)
mkdir "$D/cache"
: > "$D/srv.out"   # pre-create so the awk below can't fail on a missing file under `set -e`
# -u DFKV_LOG: the clean-shutdown check below reads INFO lines.
env -u DFKV_STORE_ENGINE -u DFKV_SLAB_WRITE -u DFKV_LOG "$BUILD/dfkv_server" \
  --dir "$D/cache" --port 0 --cap 1073741824 >>"$D/srv.out" 2>&1 &
PID=$!
# The server of the moment is asked to stop, and killed if it has not within
# two seconds: one that is blocked in an open does not act on SIGTERM.
cleanup() {
  kill $PID 2>/dev/null || true
  for _ in $(seq 1 40); do
    kill -0 $PID 2>/dev/null || break
    sleep 0.05
  done
  kill -KILL $PID 2>/dev/null || true
  rm -rf "$D"
}
trap cleanup EXIT
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

# SIGTERM: the server must report its disks clean before it exits, exit 0,
# and a restart on the same directory must be a clean start that still has
# the data.
kill -TERM $PID
rc=0; wait $PID || rc=$?
[ "$rc" = 0 ] || { echo "dfkv_server after SIGTERM: expected exit 0, got $rc"; cat "$D/srv.out"; exit 1; }
clean_line=$(grep -n 'shutdown: 1 of 1 disks clean' "$D/srv.out" | head -1 | cut -d: -f1)
done_line=$(grep -n 'dfkv_server shut down cleanly' "$D/srv.out" | head -1 | cut -d: -f1)
[ -n "$clean_line" ] && [ -n "$done_line" ] && [ "$clean_line" -lt "$done_line" ] || {
  echo "dfkv_server did not report a clean shutdown in order"
  cat "$D/srv.out"
  exit 1
}
: > "$D/srv2.out"
# This start also carries environment values that are not numbers and the log
# level as --help spells it: it must come up all the same (checked below).
env -u DFKV_STORE_ENGINE -u DFKV_SLAB_WRITE -u DFKV_SLAB_WARM_RESTART \
  DFKV_TCP_MAX_CONNS=-3 DFKV_TCP_IO_TIMEOUT_S=60s DFKV_RAM_RECLAIM_MS=abc \
  "$BUILD/dfkv_server" \
  --dir "$D/cache" --port 0 --cap 1073741824 --log WARN >>"$D/srv2.out" 2>&1 &
PID=$!
P=""
for i in $(seq 1 100); do P=$(awk '/PORT/{print $2}' "$D/srv2.out" 2>/dev/null || true); [ -n "$P" ] && break; sleep 0.05; done
[ -n "$P" ] || { echo "restarted server did not report PORT"; cat "$D/srv2.out"; exit 1; }
got=$("$BUILD/dfkvctl" --members "n=127.0.0.1:$P" --namespace "dfkv/smoke" get k)
[ "$got" = "v12345" ] || { echo "value lost across a clean restart: '$got'"; cat "$D/srv2.out"; exit 1; }
"$BUILD/dfkvctl" stat "127.0.0.1:$P" | grep -Eq '^dfkv_slab_unclean_resets_total(\{[^}]*\})? 0$' || {
  echo "restart after SIGTERM was not a clean start"
  "$BUILD/dfkvctl" stat "127.0.0.1:$P" | grep unclean || true
  cat "$D/srv2.out"
  exit 1
}
echo "clean-shutdown smoke OK"

# An invalid environment value never stops a start: it is reported once and
# the default is used. And --log takes the level in upper case.
grep -q "invalid DFKV_TCP_MAX_CONNS='-3'" "$D/srv2.out" || {
  echo "no warning for an invalid DFKV_TCP_MAX_CONNS"; cat "$D/srv2.out"; exit 1; }
grep -q "invalid DFKV_TCP_IO_TIMEOUT_S='60s'" "$D/srv2.out" || {
  echo "no warning for an invalid DFKV_TCP_IO_TIMEOUT_S"; cat "$D/srv2.out"; exit 1; }
"$BUILD/dfkvctl" stat "127.0.0.1:$P" | grep -Eq '^dfkv_tcp_max_connections(\{[^}]*\})? 512$' || {
  echo "an invalid DFKV_TCP_MAX_CONNS did not fall back to the default"
  "$BUILD/dfkvctl" stat "127.0.0.1:$P" | grep tcp_max || true; exit 1; }
if grep -q '\[INFO\]' "$D/srv2.out"; then
  echo "--log WARN still logs INFO lines"; cat "$D/srv2.out"; exit 1
fi
echo "invalid-environment smoke OK"

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
# A port outside 0..65535 was truncated to 16 bits, a negative capacity
# wrapped to 2^64-1.
rc=0; "$BUILD/dfkv_server" --dir "$D" --cap 5 --port 70000 >/dev/null 2>&1 || rc=$?
[ "$rc" = 2 ] || { echo "dfkv_server --port 70000: expected exit 2, got $rc"; exit 1; }
rc=0; "$BUILD/dfkv_server" --dir "$D" --cap -1 --port 0 >/dev/null 2>&1 || rc=$?
[ "$rc" = 2 ] || { echo "dfkv_server --cap -1: expected exit 2, got $rc"; exit 1; }
rc=0; "$BUILD/dfkv_server" --dir "$D" --cap 5 --port 0 --max-msg -1 >/dev/null 2>&1 || rc=$?
[ "$rc" = 2 ] || { echo "dfkv_server --max-msg -1: expected exit 2, got $rc"; exit 1; }
rc=0; "$BUILD/dfkv_mds" --listen 70000 >/dev/null 2>&1 || rc=$?
[ "$rc" = 2 ] || { echo "dfkv_mds --listen 70000: expected exit 2, got $rc"; exit 1; }
rc=0; "$BUILD/dfkvctl" --stat-port abc stat --all --mds 127.0.0.1:1 >/dev/null 2>&1 || rc=$?
[ "$rc" = 2 ] || { echo "dfkvctl --stat-port abc: expected exit 2, got $rc"; exit 1; }
# --mds without --id/--advertise is refused before any disk is opened.
mkdir "$D/unopened"
rc=0; timeout 10s "$BUILD/dfkv_server" --dir "$D/unopened" --cap 1073741824 --port 0 \
  --mds 127.0.0.1:1 >/dev/null 2>&1 || rc=$?
[ "$rc" = 1 ] || { echo "dfkv_server --mds without --id: expected exit 1, got $rc"; exit 1; }
[ ! -e "$D/unopened/slots.tbl" ] || { echo "--mds without --id opened the cache directory"; exit 1; }
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
echo "$out" | grep -qE '^GET_SETUP registered_arena_bytes=[1-9][0-9]* ' || {
  echo "dfkv_bench registered GET arena report missing: '$out'"
  exit 1
}
echo "$out" | grep -qE '^GET[[:space:]].*goodput [0-9.]+ GB/s.*ok=4 fails=0$' || {
  echo "dfkv_bench fail-adjusted goodput report missing: '$out'"
  exit 1
}
echo "$out" | grep -qE '^DIAG phase=GET .*ring=1 .*io_errors=0 .*completion_timeouts=0 ' || {
  echo "dfkv_bench GET diagnostics missing: '$out'"
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

# One-shot CLI failures must include native ring/transport diagnostics rather
# than only printing FAIL. Port 1 is intentionally unreachable.
rc=0
"$BUILD/dfkvctl" --members "n=127.0.0.1:1" --namespace "dfkv/smoke" \
  put unavailable value >"$D/ctl-fail.out" 2>"$D/ctl-fail.err" || rc=$?
[ "$rc" = 1 ] || {
  echo "dfkvctl failed PUT: expected exit 1, got $rc"
  exit 1
}
grep -qE 'dfkvctl put diagnostics: ring=1 io_errors=[1-9][0-9]*' \
  "$D/ctl-fail.err" || {
  echo "dfkvctl failed PUT diagnostics missing"
  cat "$D/ctl-fail.err"
  exit 1
}

# MDS must fail loud (exit 1) when etcd is unreachable, within the probe window,
# instead of running "healthy" while every registration silently fails.
rc=0; DFKV_MDS_ETCD_PROBE_MS=500 "$BUILD/dfkv_mds" --listen 0 --etcd 127.0.0.1:9 \
  >/dev/null 2>&1 || rc=$?
[ "$rc" = 1 ] || { echo "dfkv_mds bad etcd: expected exit 1, got $rc"; exit 1; }
echo "mds etcd-probe smoke OK"

# A stop signal during startup cancels the start: the server never listens,
# closes the disks it opened clean, says so in its last line and exits 0; the
# next start is a clean one and still has the data. The signal is sent when
# the server logs that it has begun to fault its RAM tier arena in: 1 GiB on
# small pages takes far longer than delivering a signal, and only what has
# been touched by then is ever used. Should this script be stalled for longer
# than the whole start all the same, the server is listening before the
# signal is sent; that is looked for first, said, and not failed. A signal
# that is merely too late for the pre-fault still cancels the start (that the
# pre-fault itself is given up is proven by a unit test).
kill $PID 2>/dev/null || true
wait $PID 2>/dev/null || true
SRV_ENV=(env -u DFKV_STORE_ENGINE -u DFKV_SLAB_WRITE -u DFKV_SLAB_WARM_RESTART -u DFKV_LOG)
mkdir "$D/c1"
: > "$D/c1.out"
"${SRV_ENV[@]}" "$BUILD/dfkv_server" --dir "$D/c1" --port 0 --cap 1073741824 >>"$D/c1.out" 2>&1 &
PID=$!
P=""
for i in $(seq 1 100); do P=$(awk '/PORT/{print $2}' "$D/c1.out" 2>/dev/null || true); [ -n "$P" ] && break; sleep 0.05; done
[ -n "$P" ] || { echo "server did not report PORT"; cat "$D/c1.out"; exit 1; }
"$BUILD/dfkvctl" --members "n=127.0.0.1:$P" --namespace "dfkv/smoke" put kept v-kept
kill -TERM $PID
rc=0; wait $PID || rc=$?
[ "$rc" = 0 ] || { echo "dfkv_server after SIGTERM: expected exit 0, got $rc"; cat "$D/c1.out"; exit 1; }

: > "$D/cancel.out"
"${SRV_ENV[@]}" DFKV_RAM_TIER=1 DFKV_RAM_TIER_BYTES=1073741824 \
  DFKV_RAM_TIER_HUGEPAGES=off DFKV_RAM_TIER_LOCK=off "$BUILD/dfkv_server" \
  --dir "$D/c1" --port 0 --cap 1073741824 >>"$D/cancel.out" 2>&1 &
PID=$!
faulting=""
for i in $(seq 1 2000); do
  grep -q 'ram tier arena: faulting in' "$D/cancel.out" 2>/dev/null && { faulting=1; break; }
  sleep 0.005
done
[ -n "$faulting" ] || { echo "the arena pre-fault did not begin"; cat "$D/cancel.out"; exit 1; }
late=""
grep -q '^PORT ' "$D/cancel.out" && late=1
kill -TERM $PID
rc=0; wait $PID || rc=$?
[ "$rc" = 0 ] || { echo "cancelled start: expected exit 0, got $rc"; cat "$D/cancel.out"; exit 1; }
if [ -n "$late" ]; then
  echo "cancelled-start smoke: the server was listening before the signal was sent (stalled runner); not checked"
else
  tail -n 1 "$D/cancel.out" | grep -q 'dfkv_server start cancelled: .*every opened disk closed clean' || {
    echo "cancelled start: the last line does not say so"
    cat "$D/cancel.out"
    exit 1
  }
  if grep -Eq '^PORT |dfkv_server listening on port' "$D/cancel.out"; then
    echo "cancelled start: the server started listening"
    cat "$D/cancel.out"
    exit 1
  fi
  grep -q 'ram tier arena pre-fault given up' "$D/cancel.out" ||
    echo "cancelled-start smoke: the signal arrived after the pre-fault"
fi

# A start is held in place without timing: the second directory's
# initialization marker is a FIFO, on which the open blocks until somebody
# opens the other end (and then fails: a FIFO is no marker). The first disk
# is open by then.
mkdir "$D/c2"
mkfifo "$D/c2/slab_initializing"

# Killed (SIGKILL) while it is held there: a disk that has been opened but is
# not serving yet has not been marked in use, so the next start is a clean
# one and has the data even with DFKV_SLAB_WARM_RESTART=off.
: > "$D/killed.out"
"${SRV_ENV[@]}" "$BUILD/dfkv_server" --dir "$D/c1,$D/c2" --port 0 --cap 2147483648 >>"$D/killed.out" 2>&1 &
PID=$!
opened=""
for i in $(seq 1 400); do
  grep -q "slab $D/c1: opened in" "$D/killed.out" 2>/dev/null && { opened=1; break; }
  sleep 0.05
done
[ -n "$opened" ] || { echo "the first disk did not open"; cat "$D/killed.out"; exit 1; }
kill -KILL $PID
wait $PID 2>/dev/null || true
: > "$D/c1k.out"
"${SRV_ENV[@]}" DFKV_SLAB_WARM_RESTART=off "$BUILD/dfkv_server" --dir "$D/c1" --port 0 --cap 1073741824 >>"$D/c1k.out" 2>&1 &
PID=$!
P=""
for i in $(seq 1 100); do P=$(awk '/PORT/{print $2}' "$D/c1k.out" 2>/dev/null || true); [ -n "$P" ] && break; sleep 0.05; done
[ -n "$P" ] || { echo "server did not report PORT after a killed start"; cat "$D/c1k.out"; exit 1; }
got=$("$BUILD/dfkvctl" --members "n=127.0.0.1:$P" --namespace "dfkv/smoke" get kept || true)
[ "$got" = "v-kept" ] || { echo "value lost by a start that was killed before it served: '$got'"; cat "$D/killed.out" "$D/c1k.out"; exit 1; }
"$BUILD/dfkvctl" stat "127.0.0.1:$P" | grep -Eq '^dfkv_slab_unclean_resets_total(\{[^}]*\})? 0$' || {
  echo "the start after a killed start reset the cache"
  cat "$D/killed.out" "$D/c1k.out"
  exit 1
}
kill -TERM $PID
rc=0; wait $PID || rc=$?
[ "$rc" = 0 ] || { echo "dfkv_server after SIGTERM: expected exit 0, got $rc"; cat "$D/c1k.out"; exit 1; }
echo "killed-start smoke OK"

# The same held start with SIGTERM and a disk that then fails to open: the
# start is cancelled, the disk that did open is closed clean, but the last
# line is a warning and the exit status 1.
: > "$D/cancel2.out"
"${SRV_ENV[@]}" "$BUILD/dfkv_server" --dir "$D/c1,$D/c2" --port 0 --cap 2147483648 >>"$D/cancel2.out" 2>&1 &
PID=$!
opened=""
for i in $(seq 1 400); do
  grep -q "slab $D/c1: opened in" "$D/cancel2.out" 2>/dev/null && { opened=1; break; }
  sleep 0.05
done
[ -n "$opened" ] || { echo "the first disk did not open"; cat "$D/cancel2.out"; exit 1; }
kill -TERM $PID
timeout 10 bash -c ': > "$1"' _ "$D/c2/slab_initializing" || {
  # Nobody was reading: the server is not where this test put it, and may be
  # blocked for good. It is killed, not waited for.
  kill -KILL $PID 2>/dev/null || true
  echo "cancelled start with a failed disk: the blocked open could not be released"
  cat "$D/cancel2.out"
  exit 1
}
rc=0; wait $PID || rc=$?
[ "$rc" = 1 ] || { echo "cancelled start with a failed disk: expected exit 1, got $rc"; cat "$D/cancel2.out"; exit 1; }
tail -n 1 "$D/cancel2.out" | grep -q 'WARN.*dfkv_server start cancelled: .*a disk failed to open or is not clean' || {
  echo "cancelled start with a failed disk: the last line does not say so"
  cat "$D/cancel2.out"
  exit 1
}
grep -q "shutdown: disk $D/c1 synced and clean" "$D/cancel2.out" || {
  echo "cancelled start with a failed disk: the disk that opened was not closed clean"
  cat "$D/cancel2.out"
  exit 1
}
if grep -Eq '^PORT |dfkv_server listening on port' "$D/cancel2.out"; then
  echo "cancelled start with a failed disk: the server started listening"
  cat "$D/cancel2.out"
  exit 1
fi
: > "$D/c1b.out"
"${SRV_ENV[@]}" "$BUILD/dfkv_server" --dir "$D/c1" --port 0 --cap 1073741824 >>"$D/c1b.out" 2>&1 &
PID=$!
P=""
for i in $(seq 1 100); do P=$(awk '/PORT/{print $2}' "$D/c1b.out" 2>/dev/null || true); [ -n "$P" ] && break; sleep 0.05; done
[ -n "$P" ] || { echo "server did not report PORT after a cancelled start"; cat "$D/c1b.out"; exit 1; }
got=$("$BUILD/dfkvctl" --members "n=127.0.0.1:$P" --namespace "dfkv/smoke" get kept)
[ "$got" = "v-kept" ] || { echo "value lost across a cancelled start: '$got'"; cat "$D/c1b.out"; exit 1; }
stat_out=$("$BUILD/dfkvctl" stat "127.0.0.1:$P")
{ echo "$stat_out" | grep -Eq '^dfkv_slab_unclean_resets_total(\{[^}]*\})? 0$' &&
  echo "$stat_out" | grep -Eq '^dfkv_slab_warm_restarts_total(\{[^}]*\})? 0$'; } || {
  echo "the start after a cancelled start was not a clean start"
  echo "$stat_out" | grep -E 'unclean|warm' || true
  cat "$D/cancel.out" "$D/cancel2.out" "$D/c1b.out"
  exit 1
}
echo "cancelled-start smoke OK"
# The server of the last check is left to the EXIT trap, like the one before.
