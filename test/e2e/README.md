# dfkv per-directory storage activation E2E

`dfkv_storage_activation.py` is the native acceptance driver for the final
candidate's unset-selector directory activation. It does not use
`/root/dfkv-dev/current`, create a `LogStore` directly, infer an engine from a
filename, or accept an injected test engine as evidence.

**As implemented here, no native acceptance run has been performed.** The
parent must supply the final candidate build and reserve the actual test
filesystem/runtime. The CPU-only controls are checker/lifecycle tests, not
release-produced storage evidence.

## Exact old release provenance

The only accepted old server/MDS/client identity is public **v2.44.0**, source
commit `ef851f2617fdf4bf2ae15747550349eb7ee28dfe`:

| Artifact | Exact binding |
|---|---|
| Public asset | `dfkv-2.44.0-linux-x86_64.tar.gz` |
| Download | `https://github.com/dingodb/DingoCache/releases/download/v2.44.0/dfkv-2.44.0-linux-x86_64.tar.gz` |
| Archive bytes | `12864294` |
| Archive SHA-256 | `da8d736baa5761d5c504110862e9453ff5d0b425be0d13f026693c7d824f945e` |
| `bin/dfkv_server` SHA-256 | `e5f4cb1e7d1c15e06e1fbf4b525e2e507fa39e3b537c25a4fc75c6d78eb21510` |
| `bin/dfkv_mds` SHA-256 | `a7c1aab276cf6e6e390cb86dab5b74613de12bde42754f633cd1aad702f72d91` |
| `lib/libdfkv.so` target SHA-256 | `90060a29ab65bbc40dca79fdb9f964668a041b7d16becf0f461924cc04acca9b` |

On this authoring host the archive was fetched from that public URL and its
size/hash checked. Extracted `VERSION`, `dfkv_server --version`, and
`dfkv_mds --version` all reported 2.44.0; ELF inspection identified little-endian
ELF64 x86-64 binaries. This was artifact inspection only, not a server run.

For a fresh reproducible fetch, choose a new artifact directory, download only
that URL, then require both checks below before passing `--release-archive`:

```bash
sha256sum dfkv-2.44.0-linux-x86_64.tar.gz
# must equal da8d736baa5761d5c504110862e9453ff5d0b425be0d13f026693c7d824f945e
stat --printf='%s\n' dfkv-2.44.0-linux-x86_64.tar.gz
# must equal 12864294
```

The driver repeats those checks, safely extracts only into its new run root,
checks ELF64/x86-64 headers, and repeats exact file hashes and all three daemon/
library version checks. `--release-dir` is also accepted for a canonical
extracted release or a validated flat directory, but its server, MDS and client
library must have the same exact hashes. A branch name, `VERSION` file alone,
or a current/running deployment is never sufficient provenance.

The released public C ABI used by the driver is the length-delimited v2 ABI in
`src/client/dfkv_c_api.h` at the commit above: `dfkv_open_v2`, `dfkv_put`,
`dfkv_get_auto`, `dfkv_batch_put_st`, `dfkv_batch_get_auto_st`,
`dfkv_remove`, `dfkv_batch_remove`, `dfkv_exist`, and `dfkv_close`.

## Parent/runtime invocation

All path values must be explicit absolute choices. `--run-root` and
`--cache-root` **must not exist**. The driver creates them once and never clears,
renames, removes, or replaces them. `--run-root` is local logs/evidence/etcd/MDS
storage and may not be under `/mnt/disk*`; only `--cache-root` selects the test
store filesystem. A `/mnt/disk*` path is not by itself proof of an NVMe claim:
the parent must supply/maintain its reservation and mount evidence.

```bash
python3 test/e2e/dfkv_storage_activation.py \
  --release-archive /tmp/dfkv-storage-activation-v2.44.0-public/dfkv-2.44.0-linux-x86_64.tar.gz \
  --candidate-build-dir /ABSOLUTE/FINAL-CANDIDATE/BUILD \
  --etcd-binary /ABSOLUTE/PINNED/etcd \
  --run-root /ABSOLUTE/FRESH-LOCAL-EVIDENCE-ROOT \
  --cache-root /mnt/diskN/ABSOLUTE/FRESH-AUTHORIZED-TEST-STORE-ROOT \
  --capacity-bytes 1073741824
```

The candidate build may be a flat build (`dfkv_server` and `libdfkv.so` at its
root) or a packaged `bin`/`lib` layout. It need not contain an MDS: every ring
uses the exact public release MDS. Prefer `--release-archive` when the validated
flat directory is unavailable. Adjust capacity only through the documented CLI;
there are no engine-selection CLI switches.

Every server/control process uses fresh loopback etcd client/peer, MDS,
data, and metrics endpoints allocated for the run. No production MDS, shared
cache directory, system service, GPU engine, `pkill`, `ps` credential scan, or
upgrade/restart/current replacement is used.

## Scenario and lifecycle map

Each row starts a separate fresh etcd + release-MDS ring and a real data-server
process. A native client is always closed before its data server is stopped.
Every subsequent open is a new data-server process with a new PID/start tick and
PGID. All daemon metrics/health listeners are down before the corresponding
stop is claimed.

| Phase | Store root | Binary and selector | Required native evidence |
|---|---|---|---|
| Release slab generation | new `cache-root/release-slab` | public v2.44.0 server/client; selector and experimental flag unset | Whole payload PUT and full-byte GET by that same release; actual `slab` metric; clean client/server close |
| Candidate old-slab open | **same** root and root device/inode | candidate; selector and experimental flag unset | Actual `slab` metric at canonical root; same payload GET; no entries/layout retype that would indicate conversion |
| Old development-LSE control | new `cache-root/release-development-lse` | public v2.44.0 server/client, explicit `log` + `DFKV_LSE_EXPERIMENTAL=1` only here | Actual `log` metric and whole payload read before close; proves the directory really is old development data |
| Candidate development refusal | **same** development root | candidate; selector and experimental flag unset | Nonzero startup exit with root and development format on one diagnostic event, no successful listener marker, no ready/engine exposure during bounded polling, and full hashes/types/symlinks/root identity unchanged |
| Fresh default write/reopen/remove | new `cache-root/fresh-default-lse` | two independent candidate server processes; selector and experimental flag unset | Actual `log` metric; public scalar direct-disk and batch PUTs, complete scalar/batch reads, second-process reads, scalar/batch removes and absence checks |

The native PUT path used by `dfkv_put` and `dfkv_batch_put_st` reaches the
server's direct-disk `DiskCacheGroup` PUT path; no direct constructor or
`new LogStore` replacement exists in the harness. Engine selection is accepted
only from the server's actual `dfkv_disk_engine_info{disk,engine}=1`, bound to
the canonical root and its real `dfkv_build_info`. No marker/file presence,
selector intent, metrics filename, first-active-engine choice, or log message
can establish the selected engine.

The expected development refusal has verdict
`EXPECTED_DEVELOPMENT_FORMAT_REFUSAL`; it is not a generic startup failure and
cannot substitute for its old-LSE positive control. The other final verdicts
are `PASS_RELEASE_SLAB_RETAINED` and
`PASS_FRESH_DEFAULT_LSE_PUBLIC_API_REOPEN_REMOVE`.

`result.json` plus logs and raw Prometheus files are preserved under the run
root. All store roots remain under the cache root. On an acceptance failure,
completed selective stages, active-stage snapshots, command rows, PID/start
ticks/PGID/actual argv (or an explicit unavailable capture for a very early
expected refusal), cleanup records and the original paths remain available.
Independent pre/post refusal hashes are in `development-before-candidate.json`
and `development-after-candidate.json`; old-slab shape/identity snapshots are
also persisted separately before and after the candidate open. Post-exit root
capture precedes diagnostic classification, so even a generic startup failure
retains both hash sets.

## Process ownership and exit status

The lifecycle helpers reuse the owned-process model from
`sglang_dfkv_full_attention.py`: new sessions, recorded PID/start tick/PGID and
actual `/proc/<pid>/cmdline`, `/proc` descendant-tree tracking, and identity
rechecks before every signal. A real process expected to refuse development data
may exit before its cmdline is readable; only that phase may record
`actual_argv_at_start=null` plus `actual_argv_captured=false` and retain the
intended argv separately, never fabricate capture. Every healthy daemon launch
still requires exact actual-argv equality. Detached descendant process groups are signalled
only by independently validated PID+start+PGID identity. Shutdown has bounded
SIGTERM/SIGKILL/reap intervals and refuses to claim success with active
children, PGID stragglers, unexpected daemon exits, or a responding health
endpoint.

- **0** — all three native scenarios and all cleanup checks passed.
- **1** — acceptance/preflight proof failed (wrong artifact/engine/payload/root,
  generic development-start failure, missing proof, or timeout).
- **3** — unclassified harness error; never a native success.
- **4** — owned-process cleanup/health-shutdown failed.

## CPU-only controls

Run from the pinned source root when the parent performs verification:

```bash
python3 test/python/test_dfkv_storage_activation_harness.py
```

The standard-library tests exercise real temporary-directory snapshots/identity,
missing and changed payload detection, partial clones, layout conversion and
symlink changes, root-bound metric parsing, release/hash/architecture bindings,
selector scrubbing, injection-before-open refusal, selective verdicts, falsified
grave-marker/generic-refusal traces, and real short-lived owned Python children
(including a detached descendant PGID). These controls neither load `libdfkv.so`
nor start etcd/dfkv and must never be presented as evidence that an older LSE
format was accepted or converted to PASS.
