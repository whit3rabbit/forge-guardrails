# Proxy Agent Notes

- Preserve no-tools passthrough as distinct from guarded tool execution.
- Forge must inspect guarded tool calls before anything executes them.
- Use `anyllm_translate` for Anthropic/OpenAI translation.
- Treat `_forge` fields, tool-choice behavior, error strings, SSE chunk shape, and tool-call IDs as compatibility-sensitive.
- Keep classifier logging opt-in, bounded where possible, and safe under concurrent requests.
- For handler changes, run `cargo test proxy::handler`; add `cargo test proxy::server` when HTTP response mapping changes.
