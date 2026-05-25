//! Stable tool-call semantic scoring API.

use std::borrow::Cow;
use std::fmt;
use std::str::FromStr;

use crate::clients::base::ToolCall;
use crate::guardrails::scoring_context::ScoringContext;

/// Classifier operating mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScorerMode {
    /// Do not run scoring or affect behavior.
    Disabled,
    /// Run scoring for telemetry only.
    #[default]
    Shadow,
    /// Allow high-confidence classifier output to request advisory nudges.
    Advisory,
    /// Allow high-confidence classifier output to block.
    Enforce,
}

impl ScorerMode {
    /// Return the stable lowercase mode name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Shadow => "shadow",
            Self::Advisory => "advisory",
            Self::Enforce => "enforce",
        }
    }
}

impl fmt::Display for ScorerMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ScorerMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "disabled" => Ok(Self::Disabled),
            "shadow" => Ok(Self::Shadow),
            "advisory" => Ok(Self::Advisory),
            "enforce" => Ok(Self::Enforce),
            other => Err(format!(
                "classifier mode must be disabled, shadow, advisory, or enforce, got '{other}'"
            )),
        }
    }
}

/// Semantic classifier label for one candidate tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolCallClass {
    /// Candidate appears valid.
    Valid,
    /// Candidate uses a known tool that is semantically wrong for the request.
    WrongToolSemantic,
    /// Candidate calls a tool when no tool is needed.
    ToolNotNeeded,
    /// Candidate should ask for clarification before tool use.
    NeedsClarification,
    /// Candidate corresponds to a deterministic guardrail failure class.
    DeterministicInvalid,
    /// Unknown label surfaced by an artifact.
    Unknown(String),
}

impl ToolCallClass {
    /// Return the stable classifier label name.
    pub fn as_label(&self) -> Cow<'_, str> {
        match self {
            Self::Valid => Cow::Borrowed("valid"),
            Self::WrongToolSemantic => Cow::Borrowed("wrong_tool_semantic"),
            Self::ToolNotNeeded => Cow::Borrowed("tool_not_needed"),
            Self::NeedsClarification => Cow::Borrowed("needs_clarification"),
            Self::DeterministicInvalid => Cow::Borrowed("deterministic_invalid"),
            Self::Unknown(label) => Cow::Borrowed(label.as_str()),
        }
    }
}

/// Classifier recommendation after thresholds and mode are applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassifierAction {
    /// Allow execution.
    Allow,
    /// Record telemetry only.
    ShadowOnly,
    /// Produce an advisory nudge.
    AdvisoryNudge,
    /// Block execution.
    Block,
}

impl ClassifierAction {
    /// Return the stable lowercase action name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Allow => "allow",
            Self::ShadowOnly => "shadow_only",
            Self::AdvisoryNudge => "advisory_nudge",
            Self::Block => "block",
        }
    }
}

/// Score for a single candidate tool call.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCallScore {
    /// Predicted semantic label.
    pub label: ToolCallClass,
    /// Softmax confidence for the selected label.
    pub confidence: f32,
    /// Raw classifier logits in label order.
    pub logits: Vec<f32>,
    /// Thresholded action recommendation.
    pub action: ClassifierAction,
    /// Classifier artifact or implementation version.
    pub model_version: String,
    /// End-to-end scoring latency in milliseconds.
    pub latency_ms: f64,
}

/// Synchronous scorer for one tool call after deterministic guardrails pass.
pub trait ToolCallScorer: Send + Sync {
    /// Score one candidate tool call.
    fn score(&self, ctx: &ScoringContext, candidate: &ToolCall) -> anyhow::Result<ToolCallScore>;
}

/// Deterministic no-op scorer used by tests and disabled configurations.
#[derive(Debug, Default)]
pub struct NoopToolCallScorer;

impl ToolCallScorer for NoopToolCallScorer {
    fn score(&self, _ctx: &ScoringContext, _candidate: &ToolCall) -> anyhow::Result<ToolCallScore> {
        Ok(ToolCallScore {
            label: ToolCallClass::Valid,
            confidence: 1.0,
            logits: Vec::new(),
            action: ClassifierAction::Allow,
            model_version: "noop".to_string(),
            latency_ms: 0.0,
        })
    }
}
