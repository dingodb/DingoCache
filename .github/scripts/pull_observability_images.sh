#!/usr/bin/env bash
# Pulls the pinned images the workflows validate the observability
# configuration with, and tags each under a fixed local name:
#
#   dfkv-ci/prometheus:ci   dfkv-ci/otel-collector:ci   dfkv-ci/tempo:ci
#
# The image references (repository and sha256 digest) are read from
# deploy/observability/docker-compose.yml, so the workflows check the
# configuration against exactly the images the compose file ships.
#
# A pull from a hosted runner fails now and then with a TLS handshake
# timeout, on any of the images. Each image is therefore tried a few times
# with a growing pause, first on the canonical registry and then on the
# registry mirror. The compose file defaults to the mirror for hosts that
# cannot reach the canonical registry; a hosted runner can, so here the
# mirror is the fallback. The digest pins the content: whichever source
# answers delivers the same image.
set -euo pipefail

compose="${1:-deploy/observability/docker-compose.yml}"
# "<registry host>:<attempts>", tried in this order.
sources="${DFKV_CI_IMAGE_SOURCES:-docker.io:3 docker.m.daocloud.io:2}"
backoff_s="${DFKV_CI_PULL_BACKOFF_S:-5}"

# The "<repository>@sha256:<digest>" the compose file pins for a repository.
pinned_ref() {
  local repository="$1" refs
  refs=$(sed -n "s#^ *image: .*/\\(${repository}@sha256:[0-9a-f]\\{64\\}\\) *\$#\\1#p" "$compose")
  if [ "$(printf '%s\n' "$refs" | grep -c .)" -ne 1 ]; then
    echo "expected exactly one pinned image for ${repository} in ${compose}" >&2
    return 1
  fi
  printf '%s\n' "$refs"
}

pull_pinned() {
  local repository="$1" tag="$2" ref source host attempts attempt
  ref=$(pinned_ref "$repository")
  for source in $sources; do
    host="${source%%:*}"
    attempts="${source##*:}"
    attempt=1
    while [ "$attempt" -le "$attempts" ]; do
      if docker pull "${host}/${ref}"; then
        docker tag "${host}/${ref}" "$tag"
        return 0
      fi
      echo "pull failed (${host}, attempt ${attempt} of ${attempts}): ${ref}" >&2
      if [ "$attempt" -lt "$attempts" ]; then
        sleep $((attempt * backoff_s))
      fi
      attempt=$((attempt + 1))
    done
  done
  echo "no registry delivered ${ref}" >&2
  return 1
}

pull_pinned prom/prometheus dfkv-ci/prometheus:ci
pull_pinned otel/opentelemetry-collector-contrib dfkv-ci/otel-collector:ci
pull_pinned grafana/tempo dfkv-ci/tempo:ci
