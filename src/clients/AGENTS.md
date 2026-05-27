# Client Development Guidelines

When modifying files under [src/clients](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/clients), adhere to the following rules to ensure compatibility and correct routing:

## Gotchas & Rules

1. **Keep TokenUsage Token-Only**:
   - The [TokenUsage](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/clients/base.rs#L144) struct should only track numeric token counts (`prompt_tokens`, `completion_tokens`, `total_tokens`).
   - Do not add provider-specific metadata (cost, cache hits, headers) to `TokenUsage`. Instead, return provider cache details via [LLMUsageDetails](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/clients/base.rs#L226) and general metadata via [LLMCallInfo](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/clients/base.rs#L196) from the `last_call_info()` / `last_usage_details()` methods.
2. **Immutable Instance Defaults**:
   - Per-request options/sampling parameters (`SamplingParams`) must override default instance settings *for that request only*.
   - Never mutate the client's internal struct fields during request execution.
3. **Assert Request Bodies in Tests**:
   - When writing client unit/integration tests, mock HTTP responses using tools like `wiremock` or mock transports.
   - You **MUST** assert the structure and fields of the outgoing JSON request body on the wire, rather than only asserting that the parsed response is correct.
4. **Wire Format Isolation**:
   - Avoid sharing assumptions or sharing helper code that leaks between Anthropic, Ollama, and OpenAI-compatible backends. Each client must handle its serialization invariants independently.
5. **No-Tools Fallbacks**:
   - Handle cases where the model generates a final response instead of executing tools, and cases where unexpected or malformed tool calls are returned.

## Testing Target

When modifying client files, run:

```bash
cargo test --package forge-guardrails --lib clients
```
