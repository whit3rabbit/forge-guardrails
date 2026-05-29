use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use anyhow::Context;
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

use crate::guardrails::classifier_artifact::{
    ClassifierModelKind, FinalResponseClassifierArtifact, Thresholds,
    DEFAULT_FINAL_RESPONSE_CLASSIFIER_REPO, FINAL_RESPONSE_SERIALIZER,
};
use crate::guardrails::scoring::{
    serialize_final_response_state_v1, ClassifierAction, FinalResponseClass, FinalResponseContext,
    FinalResponseScore, FinalResponseScorer, ScorerMode,
};

use super::cache::ScoreCache;
use super::{build_session_pool, softmax, OnnxScorerOptions, DEFAULT_ONNX_SCORE_CACHE_CAPACITY};

/// ONNX Runtime-backed scorer for final-response verifier artifacts.
pub struct OnnxFinalResponseScorer {
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
    pub(crate) cache: Mutex<ScoreCache<FinalResponseScore>>,
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
            cache: Mutex::new(ScoreCache::new(DEFAULT_ONNX_SCORE_CACHE_CAPACITY)),
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

        let score = FinalResponseScore {
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
