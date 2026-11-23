#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Fails unless every ELF given was built with DFKV_HARDENING: stack canaries
# (-fstack-protector-strong), fortified glibc calls (-D_FORTIFY_SOURCE=2) and
# full RELRO (-z relro -z now).
set -euo pipefail
[[ $# -gt 0 ]] || { echo "usage: $0 ELF..." >&2; exit 2; }
status=0
for elf in "$@"; do
  dynsyms=$(readelf --dyn-syms --wide "$elf")
  problems=()
  grep -q ' __stack_chk_fail' <<<"$dynsyms" || problems+=("no stack canaries")
  grep -qE ' __[a-z_]+_chk(@|$| )' <<<"$dynsyms" || problems+=("no fortified calls")
  readelf --program-headers --wide "$elf" | grep -q 'GNU_RELRO' || problems+=("no RELRO")
  readelf --dynamic --wide "$elf" | grep -qE '\(BIND_NOW\)|\(FLAGS(_1)?\).*\bNOW\b' ||
    problems+=("lazy binding (no -z now)")
  if [[ ${#problems[@]} -gt 0 ]]; then
    echo "$elf: ${problems[*]}" >&2
    status=1
  else
    echo "$elf: hardened"
  fi
done
exit $status
