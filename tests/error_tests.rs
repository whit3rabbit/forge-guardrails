//! Integration tests for error tests.

use forge_guardrails::error::*;
use forge_guardrails::ForgeError;

#[test]
fn unsupported_model_display() {
    let err = UnsupportedModelError::new("test-model");
    let msg = format!("{}", err);
    assert!(msg.contains("test-model"));
    assert!(msg.contains("Unsupported model"));
}

#[test]
fn tool_call_error_display() {
    let err = ToolCallError::new("no valid tool calls");
    assert_eq!(format!("{}", err), "no valid tool calls");
}

#[test]
fn tool_call_error_with_fields() {
    let err = ToolCallError::new("failed")
        .with_raw_response("raw")
        .with_cause("timeout");
    assert_eq!(err.raw_response.as_deref(), Some("raw"));
    assert_eq!(err.cause.as_deref(), Some("timeout"));
}

#[test]
fn tool_execution_error_display() {
    let err = ToolExecutionError::new("search", "network timeout");
    let msg = format!("{}", err);
    assert!(msg.contains("search"));
    assert!(msg.contains("network timeout"));
}

#[test]
fn tool_resolution_error_display() {
    let err = ToolResolutionError::new("ambiguous result");
    assert_eq!(format!("{}", err), "ambiguous result");
}

#[test]
fn tool_resolution_error_with_tool_name() {
    let err = ToolResolutionError::new("test").with_tool_name("tool_a");
    assert_eq!(err.tool_name.as_deref(), Some("tool_a"));
}

#[test]
fn workflow_cancelled_display() {
    let mut steps = indexmap::IndexMap::new();
    steps.insert("step_a".to_string(), ());
    let err = WorkflowCancelledError::new(vec!["cancelled".to_string()], steps, 3);
    let msg = format!("{}", err);
    assert!(msg.contains("3"));
    assert!(msg.contains("step_a"));
}

#[test]
fn max_iterations_display() {
    let steps = indexmap::IndexMap::new();
    let err = MaxIterationsError::new(10, steps, vec!["step_x".to_string()]);
    let msg = format!("{}", err);
    assert!(msg.contains("10"));
    assert!(msg.contains("step_x"));
}

#[test]
fn step_enforcement_display() {
    let err = StepEnforcementError::new("finish", 2, vec!["analyze".to_string()]);
    let msg = format!("{}", err);
    assert!(msg.contains("finish"));
    assert!(msg.contains("2"));
}

#[test]
fn prerequisite_error_display() {
    let err = PrerequisiteError::new("search", 3, vec!["index".to_string()]);
    let msg = format!("{}", err);
    assert!(msg.contains("search"));
    assert!(msg.contains("3"));
    assert!(msg.contains("index"));
}

#[test]
fn context_budget_exceeded_display() {
    let err = ContextBudgetExceeded::new(5000, 4096);
    let msg = format!("{}", err);
    assert!(msg.contains("5000"));
    assert!(msg.contains("4096"));
}

#[test]
fn hardware_detection_display() {
    let err = HardwareDetectionError::new("no CUDA");
    let msg = format!("{}", err);
    assert!(msg.contains("no CUDA"));
}

#[test]
fn context_discovery_display() {
    let err = ContextDiscoveryError::new("API error");
    let msg = format!("{}", err);
    assert!(msg.contains("API error"));
}

#[test]
fn budget_resolution_no_cause() {
    let err = BudgetResolutionError::new();
    let msg = format!("{}", err);
    assert!(msg.contains("No GPU detected"));
    assert!(msg.contains("no explicit budget provided"));
}

#[test]
fn budget_resolution_with_cause() {
    let err = BudgetResolutionError::new().with_cause("timeout");
    let msg = format!("{}", err);
    assert!(msg.contains("timeout"));
}

#[test]
fn backend_error_generic() {
    let err = BackendError::new(500, "server error");
    let msg = format!("{}", err);
    assert!(msg.contains("500"));
    assert!(msg.contains("server error"));
}

#[test]
fn backend_error_thinking_not_supported() {
    let err = BackendError::thinking_not_supported("granite-4.0");
    let msg = format!("{}", err);
    assert!(msg.contains("granite-4.0"));
    assert!(msg.contains("Thinking mode not supported"));
}

#[test]
fn thinking_not_supported_is_backend_error() {
    let err = ThinkingNotSupportedError::thinking_not_supported("test-model");
    let as_backend: &BackendError = &err;
    let msg = format!("{}", as_backend);
    assert!(msg.contains("test-model"));
}

#[test]
fn stream_error_default() {
    let err = StreamError::default();
    assert_eq!(format!("{}", err), "Stream ended without a final chunk");
}

#[test]
fn stream_error_custom_message() {
    let err = StreamError::new("custom message");
    assert_eq!(format!("{}", err), "custom message");
}

#[test]
fn forge_error_from_unsupported_model() {
    let inner = UnsupportedModelError::new("bad-model");
    let forge: ForgeError = inner.into();
    let msg = format!("{}", forge);
    assert!(msg.contains("bad-model"));
}

#[test]
fn forge_error_from_backend() {
    let inner = BackendError::new(503, "unavailable");
    let forge: ForgeError = inner.into();
    let msg = format!("{}", forge);
    assert!(msg.contains("503"));
}

#[test]
fn forge_error_from_stream() {
    let inner = StreamError::new("broken pipe");
    let forge: ForgeError = inner.into();
    let msg = format!("{}", forge);
    assert!(msg.contains("broken pipe"));
}

#[test]
fn tool_resolution_not_forge_error() {
    // ToolResolutionError cannot be converted to ForgeError.
    // Verify the type exists independently.
    let err = ToolResolutionError::new("test").with_tool_name("tool_a");
    assert_eq!(err.tool_name.as_deref(), Some("tool_a"));
}
