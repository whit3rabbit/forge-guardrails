# syntax=docker/dockerfile:1.7

FROM rust:bookworm AS builder

ARG ANYLLM_PROXY_VERSION=0.9.9
# An unnamed cache mount is keyed on its target path alone, so a
# multi-platform build runs both arches against one cargo registry and one
# target dir concurrently. Cargo then races itself unpacking crates
# ("failed to open .cargo-ok: File exists"). Scope every cache per platform.
ARG TARGETPLATFORM

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN --mount=type=cache,target=/usr/local/cargo/registry,id=cargo-registry-${TARGETPLATFORM},sharing=locked \
    --mount=type=cache,target=/usr/local/cargo/git,id=cargo-git-${TARGETPLATFORM},sharing=locked \
    --mount=type=cache,target=/app/target,id=forge-target-${TARGETPLATFORM},sharing=locked \
    cargo build --release --locked --bin forge-guardrails-proxy && \
    cp target/release/forge-guardrails-proxy /usr/local/bin/forge-guardrails-proxy && \
    strip /usr/local/bin/forge-guardrails-proxy

RUN --mount=type=cache,target=/usr/local/cargo/registry,id=cargo-registry-${TARGETPLATFORM},sharing=locked \
    --mount=type=cache,target=/usr/local/cargo/git,id=cargo-git-${TARGETPLATFORM},sharing=locked \
    --mount=type=cache,target=/app/target-anyllm,id=anyllm-target-${TARGETPLATFORM},sharing=locked \
    CARGO_TARGET_DIR=/app/target-anyllm cargo install anyllm_proxy --version "${ANYLLM_PROXY_VERSION}" --locked --root /usr/local && \
    strip /usr/local/bin/anyllm_proxy

FROM debian:bookworm-slim AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl libssl3 && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --system forge && \
    useradd --system --gid forge --home-dir /nonexistent --shell /usr/sbin/nologin forge && \
    mkdir -p /var/lib/forge/anyllm && \
    chown -R forge:forge /var/lib/forge

COPY --from=builder /usr/local/bin/forge-guardrails-proxy /usr/local/bin/forge-guardrails-proxy
COPY --from=builder /usr/local/bin/anyllm_proxy /usr/local/bin/anyllm_proxy
COPY docker/entrypoint.sh /usr/local/bin/forge-docker-entrypoint

ENV FORGE_HOST=0.0.0.0
ENV FORGE_PORT=8081
ENV ANYLLM_HOME=/var/lib/forge/anyllm
ENV FORGE_TOOL_OUTPUT_COMPRESSION=standard

USER forge
EXPOSE 8081
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD curl -fsS "http://127.0.0.1:${FORGE_PORT:-8081}/health" >/dev/null || exit 1
ENTRYPOINT ["forge-docker-entrypoint"]
