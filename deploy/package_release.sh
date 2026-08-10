#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Assemble the canonical self-contained Linux release bundle from a configured build.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 BUILD_DIR [OUTPUT_DIR]" >&2
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
build_dir=$(realpath "$1")
output_dir=$(realpath -m "${2:-$repo_root/release}")
version=$(<"$repo_root/VERSION")
package="dfkv-${version}-linux-x86_64"
stage="$output_dir/.stage-$package"
root="$stage/$package"

rm -rf "$stage"
mkdir -p "$root"
cmake --install "$build_dir" --prefix "$root"

# CMake installs shareable assets under share/dfkv for normal system prefixes.
# A portable tarball exposes the same repository layout at its root so relative
# documentation links and operational commands work immediately after extract.
share="$root/share/dfkv"
for directory in python integration deploy docs src test .github; do
  mv "$share/$directory" "$root/$directory"
done
for file in CMakeLists.txt Makefile Dockerfile .dockerignore README.md CHANGELOG.md VERSION LICENSE; do
  mv "$share/$file" "$root/$file"
done
rmdir "$share" "$root/share"

# Source-tree test runs must not leak generated caches into a release archive.
find "$root" -type d \( -name __pycache__ -o -name .pytest_cache -o -name .mypy_cache \) \
  -prune -exec rm -rf {} +
find "$root" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete

for tool in dfkv_server dfkv_mds dfkv_smoke dfkvctl dfkv_bench \
            dfkv_discover_smoke dfkv_gpu_dedup_repro; do
  [[ -x "$root/bin/$tool" ]] || {
    echo "missing release executable: bin/$tool" >&2
    exit 1
  }
done
for forbidden in include/gtest include/gmock lib/libgtest.a lib/libgtest_main.a \
                 lib/libgmock.a lib/libgmock_main.a; do
  [[ ! -e "$root/$forbidden" ]] || {
    echo "test-only artifact leaked into release: $forbidden" >&2
    exit 1
  }
done

major=${version%%.*}
[[ -L "$root/lib/libdfkv.so" ]] || {
  echo "missing unversioned libdfkv.so symlink" >&2
  exit 1
}
[[ "$(readlink "$root/lib/libdfkv.so")" == "libdfkv.so.$major" ]] || {
  echo "invalid libdfkv.so symlink target" >&2
  exit 1
}
[[ -L "$root/lib/libdfkv.so.$major" ]] || {
  echo "missing SONAME libdfkv.so.$major symlink" >&2
  exit 1
}
[[ "$(readlink "$root/lib/libdfkv.so.$major")" == "libdfkv.so.$version" ]] || {
  echo "invalid SONAME symlink target" >&2
  exit 1
}
[[ -f "$root/lib/libdfkv.so.$version" ]] || {
  echo "missing versioned libdfkv.so.$version" >&2
  exit 1
}

strip "$root"/bin/* "$root"/lib/libdfkv.so.*.*
mkdir -p "$output_dir"
tar -C "$stage" -czf "$output_dir/$package.tar.gz" "$package"
rm -rf "$stage"
printf '%s\n' "$output_dir/$package.tar.gz"
