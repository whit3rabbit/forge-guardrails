# Changelog

All notable changes to `forge-guardrails` are documented here.

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
