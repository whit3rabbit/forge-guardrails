//! Stable tool-call semantic scoring API.

use std::borrow::Cow;
use std::fmt;
use std::str::FromStr;

use crate::clients::base::ToolCall;
use crate::guardrails::scoring_context::{
    ScoringContext, ScoringMetadata, WorkflowStateForScoring,
};

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
    /// Candidate uses a plausible tool with semantically wrong argument values.
    WrongArgumentsSemantic,
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
            Self::WrongArgumentsSemantic => Cow::Borrowed("wrong_arguments_semantic"),
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

/// Tool result included in final-response scoring.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FinalResponseToolResult {
    /// Tool name that produced the result.
    pub tool_name: String,
    /// Text payload returned by the tool.
    pub content: String,
}

/// Complete final-response scoring context.
#[derive(Debug, Clone, PartialEq)]
pub struct FinalResponseContext {
    /// Classifier input schema version.
    pub schema_version: String,
    /// User request being satisfied.
    pub user_request: String,
    /// Current workflow state.
    pub workflow_state: WorkflowStateForScoring,
    /// Required facts or contracts when known.
    pub required_facts: Vec<String>,
    /// Ordered tool names called before the final response.
    pub tool_trace: Vec<String>,
    /// Tool results available to ground the response.
    pub tool_results: Vec<FinalResponseToolResult>,
    /// Candidate final response text.
    pub candidate_final_response: String,
    /// Optional generic eval or workflow contracts.
    pub metadata: Option<ScoringMetadata>,
}

/// Semantic classifier label for one candidate final response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinalResponseClass {
    /// Candidate final response appears valid.
    ValidFinalResponse,
    /// Candidate omits a required fact present in tool output.
    MissingToolFact,
    /// Candidate contradicts a tool result.
    ContradictsToolResult,
    /// Candidate contains an unsupported claim.
    UnsupportedClaim,
    /// Candidate fails to acknowledge missing data.
    FailedToAcknowledgeDataGap,
    /// Unknown label surfaced by an artifact.
    Unknown(String),
}

impl FinalResponseClass {
    /// Return the stable classifier label name.
    pub fn as_label(&self) -> Cow<'_, str> {
        match self {
            Self::ValidFinalResponse => Cow::Borrowed("valid_final_response"),
            Self::MissingToolFact => Cow::Borrowed("missing_tool_fact"),
            Self::ContradictsToolResult => Cow::Borrowed("contradicts_tool_result"),
            Self::UnsupportedClaim => Cow::Borrowed("unsupported_claim"),
            Self::FailedToAcknowledgeDataGap => Cow::Borrowed("failed_to_acknowledge_data_gap"),
            Self::Unknown(label) => Cow::Borrowed(label.as_str()),
        }
    }
}

/// Score for a candidate final response.
#[derive(Debug, Clone, PartialEq)]
pub struct FinalResponseScore {
    /// Predicted semantic label.
    pub label: FinalResponseClass,
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

/// Synchronous scorer for a terminal response after deterministic checks pass.
pub trait FinalResponseScorer: Send + Sync {
    /// Score one candidate final response.
    fn score(&self, ctx: &FinalResponseContext) -> anyhow::Result<FinalResponseScore>;
}

/// Serialize final-response verifier input with the published v1 format.
pub fn serialize_final_response_state_v1(ctx: &FinalResponseContext) -> String {
    let ws = &ctx.workflow_state;
    let results = ctx
        .tool_results
        .iter()
        .map(|result| format!("{}: {}", result.tool_name, json_string(&result.content)))
        .collect::<Vec<_>>()
        .join("\n");
    let metadata = ctx.metadata.as_ref();

    format!(
        "SCHEMA_VERSION:\n{}\n\nUSER_REQUEST:\n{}\n\nWORKFLOW_STATE:\nrequired_steps={}\ncompleted_steps={}\npending_steps={}\nterminal_tools={}\nrecent_errors={}\n\nREQUIRED_FACTS:\n{}\n\nTOOL_TRACE:\n{}\n\nTOOL_RESULTS:\n{}\n\nCANDIDATE_FINAL_RESPONSE:\n{}\n\nSCORING_METADATA:\nscenario_family={}\nrequires_transform={}\nrequires_synthesis={}\nrequires_all_tool_facts={}\nmust_acknowledge_missing_data={}",
        ctx.schema_version,
        ctx.user_request,
        py_list(&ws.required_steps),
        py_list(&ws.completed_steps),
        py_list(&ws.pending_steps),
        py_list(&ws.terminal_tools),
        py_list(&ws.recent_errors),
        py_list(&ctx.required_facts),
        py_list(&ctx.tool_trace),
        results,
        ctx.candidate_final_response,
        optional_json_string(metadata.and_then(|value| value.scenario_family.as_deref())),
        optional_json_bool(metadata.and_then(|value| value.requires_transform)),
        optional_json_bool(metadata.and_then(|value| value.requires_synthesis)),
        optional_json_bool(metadata.and_then(|value| value.requires_all_tool_facts)),
        optional_json_bool(metadata.and_then(|value| value.must_acknowledge_missing_data)),
    )
}

fn py_list(values: &[String]) -> String {
    if values.is_empty() {
        return "[]".to_string();
    }
    let body = values
        .iter()
        .map(|value| format!("'{}'", value.replace('\\', "\\\\").replace('\'', "\\'")))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{body}]")
}

fn json_string(value: &str) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "\"\"".to_string())
}

fn optional_json_string(value: Option<&str>) -> String {
    value.map(json_string).unwrap_or_else(|| "null".to_string())
}

fn optional_json_bool(value: Option<bool>) -> &'static str {
    match value {
        Some(true) => "true",
        Some(false) => "false",
        None => "null",
    }
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

/// Deterministic no-op final-response scorer used by tests and disabled configurations.
#[derive(Debug, Default)]
pub struct NoopFinalResponseScorer;

impl FinalResponseScorer for NoopFinalResponseScorer {
    fn score(&self, _ctx: &FinalResponseContext) -> anyhow::Result<FinalResponseScore> {
        Ok(FinalResponseScore {
            label: FinalResponseClass::ValidFinalResponse,
            confidence: 1.0,
            logits: Vec::new(),
            action: ClassifierAction::Allow,
            model_version: "noop".to_string(),
            latency_ms: 0.0,
        })
    }
}
