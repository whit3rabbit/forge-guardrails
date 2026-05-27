//! Guardrails for response validation, error tracking, and step/tool enforcement.

/// Classifier artifact metadata parsing and validation.
pub mod classifier_artifact;
/// Stateful error and retry budget tracking.
pub mod error_tracker;
#[allow(clippy::module_inception)]
/// Main guardrails engine linking validation and step enforcement.
pub mod guardrails;
/// Bounded guardrail failure memory.
pub mod history;
/// Escalating instruction nudges.
pub mod nudge;
/// Structured policy verdicts and argument validation.
pub mod policy;
/// Format and structure validation of LLM responses.
pub mod response_validator;
/// Stable semantic scoring API.
pub mod scoring;
/// Classifier input context and serializer.
pub mod scoring_context;
/// Stateful prerequisite and premature tool call enforcement.
pub mod step_enforcer;

#[cfg(feature = "classifier")]
/// ONNX Runtime-backed semantic scorer.
pub mod onnx_scorer;

pub use classifier_artifact::{
    ArtifactManifest, ClassifierArtifact, ClassifierModelKind, FinalResponseClassifierArtifact,
    LabelThreshold, LabelsFile, Thresholds, DEFAULT_CLASSIFIER_REPO, DEFAULT_CLASSIFIER_REVISION,
    DEFAULT_FINAL_RESPONSE_CLASSIFIER_REPO, DEFAULT_FINAL_RESPONSE_CLASSIFIER_REVISION,
    EXPECTED_LABELS, FINAL_RESPONSE_ARTIFACT_SCHEMA_VERSION, FINAL_RESPONSE_EXPECTED_LABELS,
    FINAL_RESPONSE_INPUT_SCHEMA_VERSION, FINAL_RESPONSE_SERIALIZER,
    FINAL_RESPONSE_THRESHOLDS_SCHEMA_VERSION, LEGACY_EXPECTED_LABELS, NEXT_INPUT_SCHEMA_VERSION,
    NEXT_SERIALIZER,
};
pub use error_tracker::ErrorTracker;
pub use guardrails::{CheckResult, GuardAction, Guardrails, TerminalTool};
pub use history::GuardrailHistory;
pub use nudge::Nudge;
pub use policy::{
    validate_tool_arguments, validate_tool_call_batch, ArgValidationError, ArgValidationKind,
    GuardrailDecision, GuardrailState, GuardrailViolation,
};
pub use response_validator::{ResponseValidator, RetryNudgeFn, ValidationResult};
pub use scoring::{
    serialize_final_response_state_v1, ClassifierAction, FinalResponseClass, FinalResponseContext,
    FinalResponseScore, FinalResponseScorer, FinalResponseToolResult, NoopFinalResponseScorer,
    NoopToolCallScorer, ScorerMode, ToolCallClass, ToolCallScore, ToolCallScorer,
};
pub use scoring_context::{
    recent_errors_from_messages, serialize_state_v1, serialize_state_v2, CandidateCallForScoring,
    ScoringContext, ScoringMetadata, ToolSpecForScoring, WorkflowStateForScoring,
};
pub use step_enforcer::{StepCheck, StepEnforcer, StepPrerequisite};

#[cfg(feature = "classifier")]
pub use onnx_scorer::{
    OnnxFinalResponseScorer, OnnxScorerOptions, OnnxToolCallScorer, MAX_ONNX_SESSION_POOL_SIZE,
};
