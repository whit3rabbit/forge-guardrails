//! Guardrails for response validation, error tracking, and step/tool enforcement.

/// Stateful error and retry budget tracking.
pub mod error_tracker;
#[allow(clippy::module_inception)]
/// Main guardrails engine linking validation and step enforcement.
pub mod guardrails;
/// Escalating instruction nudges.
pub mod nudge;
/// Format and structure validation of LLM responses.
pub mod response_validator;
/// Stateful prerequisite and premature tool call enforcement.
pub mod step_enforcer;

pub use error_tracker::ErrorTracker;
pub use guardrails::{CheckResult, GuardAction, Guardrails, TerminalTool};
pub use nudge::Nudge;
pub use response_validator::{ResponseValidator, RetryNudgeFn, ValidationResult};
pub use step_enforcer::{StepCheck, StepEnforcer, StepPrerequisite};
