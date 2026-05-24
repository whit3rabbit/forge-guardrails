# syntax=docker/dockerfile:1.7

FROM rust:bookworm AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/app/target \
    cargo build --release --locked --bin forge-guardrails-proxy && \
    cp target/release/forge-guardrails-proxy /usr/local/bin/forge-guardrails-proxy && \
    strip /usr/local/bin/forge-guardrails-proxy

FROM debian:bookworm-slim AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates libssl3 && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --system forge && \
    useradd --system --gid forge --home-dir /nonexistent --shell /usr/sbin/nologin forge

COPY --from=builder /usr/local/bin/forge-guardrails-proxy /usr/local/bin/forge-guardrails-proxy

ENV FORGE_HOST=0.0.0.0
ENV FORGE_PORT=8081

USER forge
EXPOSE 8081
ENTRYPOINT ["forge-guardrails-proxy"]
