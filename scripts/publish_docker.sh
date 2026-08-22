#!/usr/bin/env bash
set -euo pipefail

image="${IMAGE:-followthewhit3rabbit/forge-guardrails}"
# Keep in step with `version` in Cargo.toml, otherwise a current build is
# published under a stale tag alongside `latest`.
version="${VERSION:-0.1.3}"
platforms="${PLATFORMS:-linux/amd64,linux/arm64}"
builder="${BUILDER:-forge-guardrails-builder}"

if ! docker buildx inspect "$builder" >/dev/null 2>&1; then
    docker buildx create --name "$builder" --use >/dev/null
else
    docker buildx use "$builder" >/dev/null
fi

docker buildx build \
    --platform "$platforms" \
    -t "${image}:${version}" \
    -t "${image}:latest" \
    --push \
    "$@" \
    .
