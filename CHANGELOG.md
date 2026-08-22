# Changelog

All notable changes to `forge-guardrails` are documented here.

## [0.1.3] - 2026-08-22

### Added
- Added a `guardrails` Cargo feature: the pure reliability stack (response
  vocabulary, tool specs, prompts and nudges, rescue parsing, argument
  validation, step enforcement, tool policy, tool-output and schema
  compression) with no transport. `cargo build --no-default-features
  --features guardrails` resolves **29 packages against the default build's
  383**, on `serde`, `serde_json`, `thiserror`, `indexmap`, `regex-lite`,
  `anyhow`, `log` and `tracing` alone.
- Added a `proxy` Cargo feature covering everything that talks to a backend or
  serves a request: `clients`, `proxy`, `server`, the workflow runner, the
  async classifier scoring drivers, and all three binaries. It gates `axum`,
  `reqwest`, `tokio`, `sentry`, `clap`, `nix`, `async-trait`, `async-stream`,
  `futures-util`, `futures-core` and the four `anyllm_*` crates, which are now
  optional dependencies.
- Added `rust-version = "1.87"`, verified against the `guardrails` feature.
  The default feature set still needs 1.95+, and that floor belongs to the
  transport dependencies rather than to this crate's own code.

### Changed
- Moved `ToolCall`, `TextResponse` and `LLMResponse` from `clients::base` into
  the new `core::response` module, which is what makes the split expressible:
  the guardrails speak these three types and must compile without
  `clients::base`'s `futures_core` client machinery. `clients::base`
  re-exports all three, so `forge_guardrails::ToolCall` and
  `forge_guardrails::clients::base::ToolCall` both still resolve.
  `LLMResponseEnvelope` deliberately stays in `clients::base`: it carries token
  usage, rate-limit headers and the provider's raw response, which only a live
  backend can produce.
- Declared the three binaries explicitly in `Cargo.toml` with
  `required-features = ["proxy"]`. They were previously auto-discovered from
  `src/bin/*/main.rs`, and an auto-discovered binary carries no
  `required-features`, so it would be built -- and fail -- under
  `--no-default-features`.

### Compatibility
- **Additive.** `default = ["proxy", "secrets-scanner"]`, so every existing
  dependant, every public path and the `forge-guardrails-proxy` binary are
  unaffected. The full test suite (536 lib tests plus every integration
  target) passes unchanged; the `guardrails`-only build runs 135 of them
  standalone.

## [0.1.2] - 2026-06-18

### Added
- Added default-on optional `secrets-scanner` Cargo feature support.
- Added proxy process input redaction with `--redact-secrets` and `FORGE_REDACT_SECRETS=true`.
- Added scanner-backed redaction for OpenAI and Anthropic message text, prior tool-result text, and prior assistant tool-call argument payloads before upstream forwarding.

### Changed
- Tool-output compression redaction now uses `secrets_scanner` when the default feature is enabled.
- Tool-output compression keeps the legacy best-effort redaction fallback for `--no-default-features` builds.
- Tool-output compression defaults and proxy redaction controls are documented in README and user docs.

### Security
- Proxy input redaction fails closed when enabled: oversized scanner input returns `413`, and scanner setup or internal failures return `500`.
- Redaction mutates upstream request input only. It does not redact LLM responses, tool names, tool IDs, roles, model names, or tool schemas.
