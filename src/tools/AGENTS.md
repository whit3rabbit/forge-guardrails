# Tools Development Guidelines

When modifying or adding built-in tools under [src/tools](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/tools), keep the following gotchas in mind:

## Gotchas & Rules

1. **The Special Role of the Respond Tool**:
   - The `respond` tool ([respond.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/tools/respond.rs)) is unique because it signals the conclusion of a guarded workflow.
   - The proxy layer and step enforcer intercept `respond` calls to strip them out before returning the final text answer to downstream API consumers.
   - Any modifications to the schema of `respond` (e.g. its arguments or name) must be coordinated with the parsing/stripping logic in [src/proxy/proxy.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/proxy/proxy.rs) and validation logic.
2. **Deterministic Parameter Schemas**:
   - Built-in tools must register themselves with fully compliant JSON Schema parameters so they can be accurately validated by [response_validator.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/guardrails/response_validator.rs).
