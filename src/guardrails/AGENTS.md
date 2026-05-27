# Guardrails Development & Safety Guidelines

When modifying files under [src/guardrails](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/guardrails), you must respect our safety gates and model training structures:

## Gotchas & Rules

1. **Conservative Verifier Promotion**:
   - The classifier has three runtime modes: `shadow`, `advisory`, and `enforce`.
   - Never skip steps when promoting new models or label thresholds. Validate new models in `shadow` first, move to `advisory` only when replay telemetry matches performance targets, and transition to `enforce` only after explicit confirmation.
2. **Authoritative Deterministic Validation**:
   - Probabilistic predictions (e.g., `deterministic_invalid` label predictions or threshold checks) must never bypass or replace deterministic code validation.
   - If a step is blocked by deterministic rules in [step_enforcer.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/guardrails/step_enforcer.rs) or fails JSON schema validation in [response_validator.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/guardrails/response_validator.rs), those failures are authoritative.
3. **Six-Label Schema Layout**:
   - Tool-call classifier artifacts must strictly map to these six labels in this exact order:
     1. `valid`
     2. `wrong_tool_semantic`
     3. `wrong_arguments_semantic`
     4. `tool_not_needed`
     5. `needs_clarification`
     6. `deterministic_invalid`
   - Modifying this layout breaks compatibility with upstream serialized ONNX models.
4. **Recovery Errors as wrong_arguments_semantic**:
   - When a model retries/recovers a tool invocation and calls the *correct* tool but fails semantic checks, that violation must be classified as `wrong_arguments_semantic` (not `wrong_tool_semantic`).

## Testing Target

When modifying guardrails files, run:

```bash
cargo test --package forge-guardrails --lib guardrails
```
