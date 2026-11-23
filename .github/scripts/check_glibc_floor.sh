#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Fails if any ELF given needs a glibc symbol newer than GLIBC_2.34, the
# release's floor (DEPLOY.md: glibc 2.34 hosts, RHEL 9 family, run it).
# Prints the newest version each file needs.
set -euo pipefail
[[ $# -gt 0 ]] || { echo "usage: $0 ELF..." >&2; exit 2; }
status=0
for elf in "$@"; do
  versions=$(objdump -T "$elf" | grep -oE 'GLIBC_[0-9.]+' | sort -uV || true)
  echo "$elf: $(tail -1 <<<"$versions")"
  if grep -qE '^GLIBC_2\.(3[5-9]|[4-9][0-9])' <<<"$versions"; then
    echo "$elf: needs $(grep -E '^GLIBC_2\.(3[5-9]|[4-9][0-9])' <<<"$versions" | tr '\n' ' ')(floor 2.34)" >&2
    status=1
  fi
done
exit $status
