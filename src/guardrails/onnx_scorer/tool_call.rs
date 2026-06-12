use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use anyhow::Context;
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

use crate::clients::base::ToolCall;
use crate::guardrails::classifier_artifact::{
    ClassifierArtifact, ClassifierModelKind, Thresholds, DEFAULT_CLASSIFIER_REPO, NEXT_SERIALIZER,
    V3_SERIALIZER,
};
use crate::guardrails::scoring::{
    ClassifierAction, ScorerMode, ToolCallClass, ToolCallScore, ToolCallScorer,
};
use crate::guardrails::scoring_context::{
    serialize_state_v1, serialize_state_v2, serialize_state_v3, ScoringContext,
};

use super::cache::ScoreCache;
use super::{build_session_pool, softmax, OnnxScorerOptions, DEFAULT_ONNX_SCORE_CACHE_CAPACITY};

/// ONNX Runtime-backed scorer for the published tool-call verifier artifact.
pub struct OnnxToolCallScorer {
    pub(crate) tokenizer: Tokenizer,
    pub(crate) labels: Vec<String>,
    pub(crate) thresholds: Thresholds,
    pub(crate) mode: ScorerMode,
    pub(crate) model_version: String,
    pub(crate) input_schema_version: String,
    pub(crate) serializer: String,
    pub(crate) sessions: Vec<Mutex<Session>>,
    pub(crate) next_session: AtomicUsize,
    pub(crate) input_names: HashSet<String>,
    pub(crate) cache: Mutex<ScoreCache<ToolCallScore>>,
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
        let mode = mode_override.unwrap_or_default();
        artifact.validate_runtime_mode(model_kind, mode)?;
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
            mode,
            model_version: DEFAULT_CLASSIFIER_REPO.to_string(),
            input_schema_version: artifact.manifest.input_schema_version,
            serializer: artifact.manifest.serializer,
            sessions,
            next_session: AtomicUsize::new(0),
            input_names,
            cache: Mutex::new(ScoreCache::new(DEFAULT_ONNX_SCORE_CACHE_CAPACITY)),
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
            V3_SERIALIZER => serialize_state_v3(&ctx_for_artifact, candidate),
            _ => serialize_state_v1(&ctx_for_artifact, candidate),
        };
        if let Some(mut cached) = self
            .cache
            .lock()
            .map_err(|_| anyhow::anyhow!("ONNX score cache lock poisoned"))?
            .get(&text)
        {
            cached.latency_ms = start.elapsed().as_secs_f64() * 1000.0;
            return Ok(cached);
        }

        let encoding = self
            .tokenizer
            .encode(text.as_str(), true)
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

        let score = ToolCallScore {
            label,
            confidence,
            logits,
            action,
            model_version: self.model_version.clone(),
            latency_ms: start.elapsed().as_secs_f64() * 1000.0,
        };
        self.cache
            .lock()
            .map_err(|_| anyhow::anyhow!("ONNX score cache lock poisoned"))?
            .insert(text, score.clone());
        Ok(score)
    }
}
