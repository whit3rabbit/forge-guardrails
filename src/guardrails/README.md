# Guardrails & Validation Module

This module houses the deterministic validators, step sequence enforcers, and probabilistic (ONNX-driven classifier) verifiers that protect execution integrity.

## File Breakdown

- [guardrails.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/guardrails/guardrails.rs): The main entry facade [Guardrails](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/guardrails/guardrails.rs#L66) that orchestrates response checks, JSON Schema validation, and neural scoring.
- [step_enforcer.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/guardrails/step_enforcer.rs): Verifies sequence compliance, ensuring steps occur in required order and all prerequisites are satisfied.
- [response_validator.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/guardrails/response_validator.rs): Performs structural validation of tool arguments using target JSON Schemas.
- [policy.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/guardrails/policy.rs): Maps allowed and blocked tool actions based on workflow state.
- [error_tracker.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/guardrails/error_tracker.rs): Monitors execution/validation errors to halt the loop if thresholds are breached.
- [history.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/guardrails/history.rs): Tracks structural histories of guardrail events, violations, and resolutions.
- [nudge.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/guardrails/nudge.rs): Structuring of retries and helper states.
- **Neural Verification Submodule**:
  - [onnx_scorer.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/guardrails/onnx_scorer.rs): Controls the local ORT (ONNX Runtime) session for loading classifier models.
  - [classifier_artifact.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/guardrails/classifier_artifact.rs): Parses downloaded classifier models, manifests, thresholds, and configuration parameters.
  - [scoring.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/guardrails/scoring.rs): Maps label probability vectors into categorical actions (e.g., allow, nudge, fail).
  - [scoring_context.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/guardrails/scoring_context.rs): Assembles runtime execution states into serialization shapes matching verifier training inputs.
