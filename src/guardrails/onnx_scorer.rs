//! ONNX Runtime-backed tool-call scorer.

use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use anyhow::Context;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::{Tensor, TensorElementType, ValueType};
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

use crate::clients::base::ToolCall;
use crate::guardrails::classifier_artifact::{
    ClassifierArtifact, ClassifierModelKind, FinalResponseClassifierArtifact, Thresholds,
    DEFAULT_CLASSIFIER_REPO, DEFAULT_FINAL_RESPONSE_CLASSIFIER_REPO, FINAL_RESPONSE_SERIALIZER,
    NEXT_SERIALIZER,
};
use crate::guardrails::scoring::{
    serialize_final_response_state_v1, ClassifierAction, FinalResponseClass, FinalResponseContext,
    FinalResponseScore, FinalResponseScorer, ScorerMode, ToolCallClass, ToolCallScore,
    ToolCallScorer,
};
use crate::guardrails::scoring_context::{serialize_state_v1, serialize_state_v2, ScoringContext};

/// Maximum number of ONNX sessions a scorer may hold.
pub const MAX_ONNX_SESSION_POOL_SIZE: usize = 4;

/// Runtime options for ONNX-backed semantic scorers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OnnxScorerOptions {
    /// Number of ONNX Runtime sessions to load for concurrent scoring.
    pub session_pool_size: usize,
    /// ONNX Runtime intra-op thread count per session.
    pub intra_threads: usize,
}

impl Default for OnnxScorerOptions {
    fn default() -> Self {
        Self {
            session_pool_size: 1,
            intra_threads: 1,
        }
    }
}

impl OnnxScorerOptions {
    /// Validate that option values stay inside bounded runtime limits.
    pub fn validate(self) -> anyhow::Result<Self> {
        anyhow::ensure!(
            (1..=MAX_ONNX_SESSION_POOL_SIZE).contains(&self.session_pool_size),
            "ONNX session pool size must be between 1 and {}, got {}",
            MAX_ONNX_SESSION_POOL_SIZE,
            self.session_pool_size
        );
        anyhow::ensure!(
            self.intra_threads > 0,
            "ONNX intra-op thread count must be positive"
        );
        Ok(self)
    }
}

/// ONNX Runtime-backed scorer for the published tool-call verifier artifact.
pub struct OnnxToolCallScorer {
    tokenizer: Tokenizer,
    labels: Vec<String>,
    thresholds: Thresholds,
    mode: ScorerMode,
    model_version: String,
    input_schema_version: String,
    serializer: String,
    sessions: Vec<Mutex<Session>>,
    next_session: AtomicUsize,
    input_names: HashSet<String>,
}

impl OnnxToolCallScorer {
    /// Load a quantized classifier artifact from a local directory.
    pub fn from_dir(
        path: impl AsRef<Path>,
        mode_override: Option<ScorerMode>,
    ) -> anyhow::Result<Self> {
        Self::from_dir_with_model(path, mode_override, ClassifierModelKind::Quantized)
    }

    /// Load a classifier artifact from a local directory with explicit model selection.
    pub fn from_dir_with_model(
        path: impl AsRef<Path>,
        mode_override: Option<ScorerMode>,
        model_kind: ClassifierModelKind,
    ) -> anyhow::Result<Self> {
        Self::from_dir_with_model_and_options(
            path,
            mode_override,
            model_kind,
            OnnxScorerOptions::default(),
        )
    }

    /// Load a classifier artifact with explicit model and runtime options.
    pub fn from_dir_with_model_and_options(
        path: impl AsRef<Path>,
        mode_override: Option<ScorerMode>,
        model_kind: ClassifierModelKind,
        options: OnnxScorerOptions,
    ) -> anyhow::Result<Self> {
        let options = options.validate()?;
        let artifact = ClassifierArtifact::from_dir(path)?;
        let tokenizer_path = artifact.dir.join("tokenizer.json");
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|err| anyhow::anyhow!("failed to load {}: {err}", tokenizer_path.display()))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: artifact.manifest.max_length,
                ..Default::default()
            }))
            .map_err(|err| anyhow::anyhow!("failed to set tokenizer truncation: {err}"))?;
        tokenizer.with_padding(Some(PaddingParams {
            pad_to_multiple_of: Some(8),
            ..Default::default()
        }));

        let model_path = artifact.model_path(model_kind);
        let (sessions, input_names) =
            build_session_pool(&model_path, options, artifact.labels.labels.len())?;

        Ok(Self {
            tokenizer,
            labels: artifact.labels.labels,
            thresholds: artifact.thresholds,
            mode: mode_override.unwrap_or_default(),
            model_version: DEFAULT_CLASSIFIER_REPO.to_string(),
            input_schema_version: artifact.manifest.input_schema_version,
            serializer: artifact.manifest.serializer,
            sessions,
            next_session: AtomicUsize::new(0),
            input_names,
        })
    }

    fn classify_label(&self, id: usize) -> ToolCallClass {
        match self.labels.get(id).map(String::as_str) {
            Some("valid") => ToolCallClass::Valid,
            Some("wrong_tool_semantic") => ToolCallClass::WrongToolSemantic,
            Some("wrong_arguments_semantic") => ToolCallClass::WrongArgumentsSemantic,
            Some("tool_not_needed") => ToolCallClass::ToolNotNeeded,
            Some("needs_clarification") => ToolCallClass::NeedsClarification,
            Some("deterministic_invalid") => ToolCallClass::DeterministicInvalid,
            Some(other) => ToolCallClass::Unknown(other.to_string()),
            None => ToolCallClass::Unknown(format!("label_id_{id}")),
        }
    }

    fn action_for(&self, label: &ToolCallClass, confidence: f32) -> ClassifierAction {
        if matches!(self.mode, ScorerMode::Disabled) {
            return ClassifierAction::Allow;
        }
        if matches!(label, ToolCallClass::Valid) {
            return ClassifierAction::Allow;
        }
        if matches!(label, ToolCallClass::DeterministicInvalid) {
            return ClassifierAction::ShadowOnly;
        }

        let threshold = self.thresholds.for_label(label);
        match self.mode {
            ScorerMode::Disabled => ClassifierAction::Allow,
            ScorerMode::Shadow => ClassifierAction::ShadowOnly,
            ScorerMode::Advisory => {
                if confidence >= threshold.advisory_min_confidence {
                    ClassifierAction::AdvisoryNudge
                } else {
                    ClassifierAction::ShadowOnly
                }
            }
            ScorerMode::Enforce => {
                if confidence >= threshold.enforce_min_confidence {
                    ClassifierAction::Block
                } else if confidence >= threshold.advisory_min_confidence {
                    ClassifierAction::AdvisoryNudge
                } else {
                    ClassifierAction::ShadowOnly
                }
            }
        }
    }
}

impl ToolCallScorer for OnnxToolCallScorer {
    fn score(&self, ctx: &ScoringContext, candidate: &ToolCall) -> anyhow::Result<ToolCallScore> {
        let start = Instant::now();
        let mut ctx_for_artifact = ctx.clone();
        ctx_for_artifact.schema_version = self.input_schema_version.clone();
        let text = match self.serializer.as_str() {
            NEXT_SERIALIZER => serialize_state_v2(&ctx_for_artifact, candidate),
            _ => serialize_state_v1(&ctx_for_artifact, candidate),
        };
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|err| anyhow::anyhow!("tokenization failed: {err}"))?;

        let input_ids = encoding
            .get_ids()
            .iter()
            .map(|&value| value as i64)
            .collect::<Vec<_>>();
        let attention_mask = encoding
            .get_attention_mask()
            .iter()
            .map(|&value| value as i64)
            .collect::<Vec<_>>();
        anyhow::ensure!(!input_ids.is_empty(), "tokenizer produced empty input_ids");
        anyhow::ensure!(
            input_ids.len() == attention_mask.len(),
            "input_ids and attention_mask lengths differ"
        );

        let seq_len = input_ids.len();
        let mut inputs = ort::inputs![
            "input_ids" => Tensor::from_array(([1usize, seq_len], input_ids.into_boxed_slice()))?,
            "attention_mask" => Tensor::from_array(([1usize, seq_len], attention_mask.into_boxed_slice()))?,
        ];
        if self.input_names.contains("token_type_ids") {
            inputs.push((
                "token_type_ids".into(),
                Tensor::from_array(([1usize, seq_len], vec![0_i64; seq_len].into_boxed_slice()))?
                    .into(),
            ));
        }

        let index = self.next_session.fetch_add(1, Ordering::Relaxed) % self.sessions.len();
        let mut session = self.sessions[index]
            .lock()
            .map_err(|_| anyhow::anyhow!("ONNX session lock poisoned"))?;
        let outputs = session.run(inputs)?;
        let output = outputs.get("logits").unwrap_or(&outputs[0]);
        let (shape, data) = output.try_extract_tensor::<f32>()?;
        anyhow::ensure!(
            shape.num_elements() == self.labels.len(),
            "classifier logits shape {} has {} values; expected {}",
            shape,
            shape.num_elements(),
            self.labels.len()
        );
        let logits = data.to_vec();
        let probs = softmax(&logits);
        let (best_id, confidence) = probs
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .context("empty classifier logits")?;
        let label = self.classify_label(best_id);
        let action = self.action_for(&label, confidence);

        Ok(ToolCallScore {
            label,
            confidence,
            logits,
            action,
            model_version: self.model_version.clone(),
            latency_ms: start.elapsed().as_secs_f64() * 1000.0,
        })
    }
}

/// ONNX Runtime-backed scorer for final-response verifier artifacts.
pub struct OnnxFinalResponseScorer {
    tokenizer: Tokenizer,
    labels: Vec<String>,
    thresholds: Thresholds,
    mode: ScorerMode,
    model_version: String,
    input_schema_version: String,
    serializer: String,
    sessions: Vec<Mutex<Session>>,
    next_session: AtomicUsize,
    input_names: HashSet<String>,
}

impl OnnxFinalResponseScorer {
    /// Prepare a final-response scorer from a local artifact directory.
    pub fn from_dir(
        path: impl AsRef<Path>,
        mode_override: Option<ScorerMode>,
    ) -> anyhow::Result<Self> {
        Self::from_dir_with_model(path, mode_override, ClassifierModelKind::Quantized)
    }

    /// Prepare a final-response scorer from a local artifact directory with model selection.
    pub fn from_dir_with_model(
        path: impl AsRef<Path>,
        mode_override: Option<ScorerMode>,
        model_kind: ClassifierModelKind,
    ) -> anyhow::Result<Self> {
        Self::from_dir_with_model_and_options(
            path,
            mode_override,
            model_kind,
            OnnxScorerOptions::default(),
        )
    }

    /// Prepare a final-response scorer with explicit model and runtime options.
    pub fn from_dir_with_model_and_options(
        path: impl AsRef<Path>,
        mode_override: Option<ScorerMode>,
        model_kind: ClassifierModelKind,
        options: OnnxScorerOptions,
    ) -> anyhow::Result<Self> {
        let options = options.validate()?;
        let artifact = FinalResponseClassifierArtifact::from_dir(path)?;
        anyhow::ensure!(
            artifact.manifest.serializer == FINAL_RESPONSE_SERIALIZER,
            "unsupported final-response serializer '{}'",
            artifact.manifest.serializer
        );
        let tokenizer_path = artifact.dir.join("tokenizer.json");
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|err| anyhow::anyhow!("failed to load {}: {err}", tokenizer_path.display()))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: artifact.manifest.max_length,
                ..Default::default()
            }))
            .map_err(|err| anyhow::anyhow!("failed to set tokenizer truncation: {err}"))?;
        tokenizer.with_padding(Some(PaddingParams {
            pad_to_multiple_of: Some(8),
            ..Default::default()
        }));

        let model_path = artifact.model_path(model_kind);
        let (sessions, input_names) =
            build_session_pool(&model_path, options, artifact.labels.labels.len())?;

        Ok(Self {
            tokenizer,
            labels: artifact.labels.labels,
            thresholds: artifact.thresholds,
            mode: mode_override.unwrap_or_default(),
            model_version: DEFAULT_FINAL_RESPONSE_CLASSIFIER_REPO.to_string(),
            input_schema_version: artifact.manifest.input_schema_version,
            serializer: artifact.manifest.serializer,
            sessions,
            next_session: AtomicUsize::new(0),
            input_names,
        })
    }

    /// Return the configured scorer mode.
    pub fn mode(&self) -> ScorerMode {
        self.mode
    }

    fn classify_label(&self, id: usize) -> FinalResponseClass {
        match self.labels.get(id).map(String::as_str) {
            Some("valid_final_response") => FinalResponseClass::ValidFinalResponse,
            Some("missing_tool_fact") => FinalResponseClass::MissingToolFact,
            Some("contradicts_tool_result") => FinalResponseClass::ContradictsToolResult,
            Some("unsupported_claim") => FinalResponseClass::UnsupportedClaim,
            Some("failed_to_acknowledge_data_gap") => {
                FinalResponseClass::FailedToAcknowledgeDataGap
            }
            Some(other) => FinalResponseClass::Unknown(other.to_string()),
            None => FinalResponseClass::Unknown(format!("label_id_{id}")),
        }
    }

    fn action_for(&self, label: &FinalResponseClass, confidence: f32) -> ClassifierAction {
        if matches!(self.mode, ScorerMode::Disabled) {
            return ClassifierAction::Allow;
        }
        if matches!(label, FinalResponseClass::ValidFinalResponse) {
            return ClassifierAction::Allow;
        }

        let threshold = self.thresholds.for_final_response_label(label);
        match self.mode {
            ScorerMode::Disabled => ClassifierAction::Allow,
            ScorerMode::Shadow => ClassifierAction::ShadowOnly,
            ScorerMode::Advisory => {
                if confidence >= threshold.advisory_min_confidence {
                    ClassifierAction::AdvisoryNudge
                } else {
                    ClassifierAction::ShadowOnly
                }
            }
            ScorerMode::Enforce => {
                if confidence >= threshold.enforce_min_confidence {
                    ClassifierAction::Block
                } else if confidence >= threshold.advisory_min_confidence {
                    ClassifierAction::AdvisoryNudge
                } else {
                    ClassifierAction::ShadowOnly
                }
            }
        }
    }
}

impl FinalResponseScorer for OnnxFinalResponseScorer {
    fn score(&self, ctx: &FinalResponseContext) -> anyhow::Result<FinalResponseScore> {
        let start = Instant::now();
        let mut ctx_for_artifact = ctx.clone();
        ctx_for_artifact.schema_version = self.input_schema_version.clone();
        let text = match self.serializer.as_str() {
            FINAL_RESPONSE_SERIALIZER => serialize_final_response_state_v1(&ctx_for_artifact),
            other => anyhow::bail!("unsupported final-response serializer '{other}'"),
        };
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|err| anyhow::anyhow!("tokenization failed: {err}"))?;

        let input_ids = encoding
            .get_ids()
            .iter()
            .map(|&value| value as i64)
            .collect::<Vec<_>>();
        let attention_mask = encoding
            .get_attention_mask()
            .iter()
            .map(|&value| value as i64)
            .collect::<Vec<_>>();
        anyhow::ensure!(!input_ids.is_empty(), "tokenizer produced empty input_ids");
        anyhow::ensure!(
            input_ids.len() == attention_mask.len(),
            "input_ids and attention_mask lengths differ"
        );

        let seq_len = input_ids.len();
        let mut inputs = ort::inputs![
            "input_ids" => Tensor::from_array(([1usize, seq_len], input_ids.into_boxed_slice()))?,
            "attention_mask" => Tensor::from_array(([1usize, seq_len], attention_mask.into_boxed_slice()))?,
        ];
        if self.input_names.contains("token_type_ids") {
            inputs.push((
                "token_type_ids".into(),
                Tensor::from_array(([1usize, seq_len], vec![0_i64; seq_len].into_boxed_slice()))?
                    .into(),
            ));
        }

        let index = self.next_session.fetch_add(1, Ordering::Relaxed) % self.sessions.len();
        let mut session = self.sessions[index]
            .lock()
            .map_err(|_| anyhow::anyhow!("ONNX session lock poisoned"))?;
        let outputs = session.run(inputs)?;
        let output = outputs.get("logits").unwrap_or(&outputs[0]);
        let (shape, data) = output.try_extract_tensor::<f32>()?;
        anyhow::ensure!(
            shape.num_elements() == self.labels.len(),
            "final-response classifier logits shape {} has {} values; expected {}",
            shape,
            shape.num_elements(),
            self.labels.len()
        );
        let logits = data.to_vec();
        let probs = softmax(&logits);
        let (best_id, confidence) = probs
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .context("empty final-response classifier logits")?;
        let label = self.classify_label(best_id);
        let action = self.action_for(&label, confidence);

        Ok(FinalResponseScore {
            label,
            confidence,
            logits,
            action,
            model_version: self.model_version.clone(),
            latency_ms: start.elapsed().as_secs_f64() * 1000.0,
        })
    }
}

fn build_session_pool(
    model_path: &Path,
    options: OnnxScorerOptions,
    label_count: usize,
) -> anyhow::Result<(Vec<Mutex<Session>>, HashSet<String>)> {
    let mut sessions = Vec::with_capacity(options.session_pool_size);
    let mut expected_input_names = None;
    for _ in 0..options.session_pool_size {
        let session = build_session(model_path, options.intra_threads)?;
        let input_names = validate_session_inputs(&session)?;
        validate_session_outputs(&session, label_count)?;
        if let Some(expected) = &expected_input_names {
            anyhow::ensure!(
                expected == &input_names,
                "ONNX session pool input names differ across sessions"
            );
        } else {
            expected_input_names = Some(input_names.clone());
        }
        sessions.push(Mutex::new(session));
    }
    let input_names = expected_input_names.context("ONNX session pool is empty")?;
    Ok((sessions, input_names))
}

fn build_session(model_path: &Path, intra_threads: usize) -> anyhow::Result<Session> {
    let mut builder = Session::builder()?;
    builder = builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|err| anyhow::anyhow!("failed to set ONNX optimization level: {err}"))?;
    builder = builder
        .with_intra_threads(intra_threads)
        .map_err(|err| anyhow::anyhow!("failed to set ONNX intra-op threads: {err}"))?;
    builder
        .commit_from_file(model_path)
        .with_context(|| format!("failed to load ONNX model {}", model_path.display()))
}

fn validate_session_inputs(session: &Session) -> anyhow::Result<HashSet<String>> {
    let mut names = HashSet::new();
    for input in session.inputs() {
        let name = input.name().to_string();
        anyhow::ensure!(
            input.dtype().tensor_type() == Some(TensorElementType::Int64),
            "classifier input '{}' must be int64 tensor, got {}",
            name,
            input.dtype()
        );
        match name.as_str() {
            "input_ids" | "attention_mask" | "token_type_ids" => {
                names.insert(name);
            }
            other => {
                anyhow::bail!("unsupported classifier ONNX input '{other}'");
            }
        }
    }
    anyhow::ensure!(
        names.contains("input_ids"),
        "classifier ONNX model is missing input_ids"
    );
    anyhow::ensure!(
        names.contains("attention_mask"),
        "classifier ONNX model is missing attention_mask"
    );
    Ok(names)
}

fn validate_session_outputs(session: &Session, label_count: usize) -> anyhow::Result<()> {
    anyhow::ensure!(
        !session.outputs().is_empty(),
        "classifier ONNX model has no outputs"
    );
    let output = session
        .outputs()
        .iter()
        .find(|output| output.name() == "logits")
        .unwrap_or(&session.outputs()[0]);
    anyhow::ensure!(
        matches!(
            output.dtype(),
            ValueType::Tensor {
                ty: TensorElementType::Float32,
                ..
            }
        ),
        "classifier output '{}' must be float32 tensor, got {}",
        output.name(),
        output.dtype()
    );
    if let Some(shape) = output.dtype().tensor_shape() {
        if let Some(&last_dim) = shape.last() {
            anyhow::ensure!(
                last_dim < 0 || last_dim as usize == label_count,
                "classifier output '{}' last dimension must be {}, got {}",
                output.name(),
                label_count,
                last_dim
            );
        }
    }
    Ok(())
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps = logits
        .iter()
        .map(|value| (*value - max).exp())
        .collect::<Vec<_>>();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        return vec![0.0; logits.len()];
    }
    exps.into_iter().map(|value| value / sum).collect()
}

#[cfg(test)]
mod tests {
    use indexmap::IndexMap;
    use serde_json::Value;

    use super::{softmax, OnnxToolCallScorer};
    use crate::clients::base::ToolCall;
    use crate::guardrails::scoring::{ScorerMode, ToolCallScorer};
    use crate::guardrails::scoring_context::ScoringContext;

    #[test]
    fn softmax_sums_to_one() {
        let probs = softmax(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.00001);
    }

    #[test]
    fn onnx_fixture_scores_without_panic_when_test_dir_is_set() {
        let Ok(dir) = std::env::var("FORGE_CLASSIFIER_TEST_DIR") else {
            return;
        };
        let scorer =
            OnnxToolCallScorer::from_dir(dir.as_str(), Some(ScorerMode::Shadow)).expect("scorer");
        let fixture_path = std::path::Path::new(&dir).join("serializer_fixture.json");
        let fixture: Value = serde_json::from_str(
            &std::fs::read_to_string(&fixture_path).expect("serializer fixture"),
        )
        .expect("serializer fixture json");
        let ctx = scoring_context_from_fixture(&fixture);
        let candidate = candidate_from_fixture(&fixture);
        let expected_logits = scorer.labels.len();
        let score = scorer.score(&ctx, &candidate).expect("score");

        assert_eq!(score.logits.len(), expected_logits);
    }

    fn scoring_context_from_fixture(value: &Value) -> ScoringContext {
        ScoringContext {
            schema_version: value["input"]["schema_version"]
                .as_str()
                .expect("schema_version")
                .to_string(),
            user_request: value["input"]["user_request"]
                .as_str()
                .expect("user_request")
                .to_string(),
            workflow_state: serde_json::from_value(value["input"]["workflow_state"].clone())
                .expect("workflow_state"),
            available_tools: serde_json::from_value(value["input"]["available_tools"].clone())
                .expect("available_tools"),
            metadata: None,
        }
    }

    fn candidate_from_fixture(value: &Value) -> ToolCall {
        let candidate = &value["input"]["candidate_call"];
        let args = candidate["arguments"]
            .as_object()
            .expect("arguments object")
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect::<IndexMap<_, _>>();
        ToolCall::new(candidate["name"].as_str().expect("name"), args)
    }
}
