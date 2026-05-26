//! Classifier artifact metadata parsing and validation.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{Context, Result as AnyResult};
use serde::Deserialize;

use crate::guardrails::scoring::{FinalResponseClass, ToolCallClass};

/// Default Hugging Face model repository for the tool-call verifier.
pub const DEFAULT_CLASSIFIER_REPO: &str = "cowWhySo/toolcall-verifier-classifier-production";
/// Pinned Hugging Face revision used by the downloader unless overridden.
pub const DEFAULT_CLASSIFIER_REVISION: &str = "662e7783c0aa25af9d8e8b74c16ef67e8bb45f03";
/// Expected classifier artifact schema version.
pub const EXPECTED_ARTIFACT_SCHEMA_VERSION: &str = "toolcall-verifier-artifact/v1";
/// Expected classifier input schema version.
pub const EXPECTED_INPUT_SCHEMA_VERSION: &str = "toolcall-verifier-input/v1";
/// Metadata-aware classifier input schema version.
pub const NEXT_INPUT_SCHEMA_VERSION: &str = "toolcall-verifier-input/v2";
/// Expected classifier serializer name.
pub const EXPECTED_SERIALIZER: &str = "serialize_state_v1";
/// Metadata-aware classifier serializer name.
pub const NEXT_SERIALIZER: &str = "serialize_state_v2";
/// Expected classifier thresholds schema version.
pub const EXPECTED_THRESHOLDS_SCHEMA_VERSION: &str = "toolcall-verifier-thresholds/v1";
/// Expected final-response artifact schema version.
pub const FINAL_RESPONSE_ARTIFACT_SCHEMA_VERSION: &str = "final-response-verifier-artifact/v1";
/// Expected final-response input schema version.
pub const FINAL_RESPONSE_INPUT_SCHEMA_VERSION: &str = "final-response-verifier-input/v1";
/// Expected final-response serializer name.
pub const FINAL_RESPONSE_SERIALIZER: &str = "serialize_final_response_state_v1";
/// Expected final-response thresholds schema version.
pub const FINAL_RESPONSE_THRESHOLDS_SCHEMA_VERSION: &str = "final-response-verifier-thresholds/v1";
/// Labels expected by the legacy production ONNX classifier.
pub const LEGACY_EXPECTED_LABELS: [&str; 5] = [
    "valid",
    "wrong_tool_semantic",
    "tool_not_needed",
    "needs_clarification",
    "deterministic_invalid",
];
/// Labels expected by the next production ONNX classifier.
pub const EXPECTED_LABELS: [&str; 6] = [
    "valid",
    "wrong_tool_semantic",
    "wrong_arguments_semantic",
    "tool_not_needed",
    "needs_clarification",
    "deterministic_invalid",
];
/// Labels expected by the final-response verifier.
pub const FINAL_RESPONSE_EXPECTED_LABELS: [&str; 5] = [
    "valid_final_response",
    "missing_tool_fact",
    "contradicts_tool_result",
    "unsupported_claim",
    "failed_to_acknowledge_data_gap",
];

/// ONNX model file selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ClassifierModelKind {
    /// Use the quantized ONNX model.
    #[default]
    Quantized,
    /// Use the full-size ONNX model.
    Full,
}

impl ClassifierModelKind {
    /// Return the stable lowercase model kind.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Quantized => "quantized",
            Self::Full => "full",
        }
    }
}

impl FromStr for ClassifierModelKind {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "quantized" => Ok(Self::Quantized),
            "full" => Ok(Self::Full),
            other => Err(format!(
                "classifier model must be quantized or full, got '{other}'"
            )),
        }
    }
}

/// Classifier artifact manifest.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct ArtifactManifest {
    /// Artifact schema version.
    pub artifact_schema_version: String,
    /// Model kind.
    #[serde(default)]
    pub model_kind: String,
    /// Base model identifier.
    #[serde(default)]
    pub base_model: String,
    /// Label mode.
    #[serde(default)]
    pub label_mode: String,
    /// Input schema version.
    pub input_schema_version: String,
    /// Serializer name.
    pub serializer: String,
    /// Maximum tokenizer sequence length.
    pub max_length: usize,
    /// Full ONNX filename.
    pub onnx_file: String,
    /// Quantized ONNX filename.
    pub quantized_onnx_file: String,
    /// Production labels in model-output order.
    #[serde(default)]
    pub labels: Vec<String>,
    /// Artifact creation timestamp, if present.
    #[serde(default)]
    pub created_unix: Option<i64>,
}

impl ArtifactManifest {
    /// Validate manifest fields required by the Rust scorer.
    pub fn validate(&self) -> AnyResult<()> {
        self.validate_tool_call()
    }

    /// Validate manifest fields required by the tool-call scorer.
    pub fn validate_tool_call(&self) -> AnyResult<()> {
        anyhow::ensure!(
            self.artifact_schema_version == EXPECTED_ARTIFACT_SCHEMA_VERSION,
            "unsupported classifier artifact schema '{}'",
            self.artifact_schema_version
        );
        anyhow::ensure!(
            self.input_schema_version == EXPECTED_INPUT_SCHEMA_VERSION
                || self.input_schema_version == NEXT_INPUT_SCHEMA_VERSION,
            "unsupported classifier input schema '{}'",
            self.input_schema_version
        );
        validate_tool_call_serializer_pair(&self.input_schema_version, &self.serializer)?;
        if !self.labels.is_empty() {
            validate_supported_label_order(&self.labels)?;
        }
        anyhow::ensure!(
            self.max_length > 0,
            "classifier max_length must be positive"
        );
        Ok(())
    }

    /// Validate manifest fields required by the final-response scorer.
    pub fn validate_final_response(&self) -> AnyResult<()> {
        anyhow::ensure!(
            self.artifact_schema_version == FINAL_RESPONSE_ARTIFACT_SCHEMA_VERSION,
            "unsupported final-response artifact schema '{}'",
            self.artifact_schema_version
        );
        anyhow::ensure!(
            self.input_schema_version == FINAL_RESPONSE_INPUT_SCHEMA_VERSION,
            "unsupported final-response input schema '{}'",
            self.input_schema_version
        );
        anyhow::ensure!(
            self.serializer == FINAL_RESPONSE_SERIALIZER,
            "unsupported final-response serializer '{}'",
            self.serializer
        );
        if !self.labels.is_empty() {
            validate_final_response_label_order(&self.labels)?;
        }
        anyhow::ensure!(
            self.max_length > 0,
            "final-response classifier max_length must be positive"
        );
        Ok(())
    }
}

/// Per-label advisory/enforcement thresholds.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct LabelThreshold {
    /// Published threshold action policy.
    pub action: String,
    /// Minimum confidence for advisory behavior.
    pub advisory_min_confidence: f32,
    /// Minimum confidence for enforcement behavior.
    pub enforce_min_confidence: f32,
}

/// Threshold file for classifier actions.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct Thresholds {
    /// Threshold schema version.
    pub schema_version: String,
    /// Published operating mode.
    pub mode: String,
    /// Default action when no label-specific threshold exists.
    pub default_action: String,
    /// Per-label thresholds.
    pub labels: HashMap<String, LabelThreshold>,
}

impl Thresholds {
    /// Validate threshold file fields required by the Rust scorer.
    pub fn validate(&self) -> AnyResult<()> {
        self.validate_tool_call()
    }

    /// Validate threshold file fields required by the tool-call scorer.
    pub fn validate_tool_call(&self) -> AnyResult<()> {
        anyhow::ensure!(
            self.schema_version == EXPECTED_THRESHOLDS_SCHEMA_VERSION,
            "unsupported classifier thresholds schema '{}'",
            self.schema_version
        );
        for label in LEGACY_EXPECTED_LABELS {
            let threshold = self
                .labels
                .get(label)
                .ok_or_else(|| anyhow::anyhow!("missing classifier threshold for '{label}'"))?;
            anyhow::ensure!(
                threshold.advisory_min_confidence.is_finite()
                    && threshold.enforce_min_confidence.is_finite(),
                "classifier thresholds for '{label}' must be finite"
            );
        }
        if let Some(threshold) = self.labels.get("wrong_arguments_semantic") {
            anyhow::ensure!(
                threshold.advisory_min_confidence.is_finite()
                    && threshold.enforce_min_confidence.is_finite(),
                "classifier thresholds for 'wrong_arguments_semantic' must be finite"
            );
        }
        Ok(())
    }

    /// Validate threshold file fields against an already-validated label order.
    pub fn validate_tool_call_for_labels(&self, labels: &[String]) -> AnyResult<()> {
        self.validate_tool_call()?;
        for label in labels {
            let threshold = self
                .labels
                .get(label.as_str())
                .ok_or_else(|| anyhow::anyhow!("missing classifier threshold for '{label}'"))?;
            anyhow::ensure!(
                threshold.advisory_min_confidence.is_finite()
                    && threshold.enforce_min_confidence.is_finite(),
                "classifier thresholds for '{label}' must be finite"
            );
        }
        Ok(())
    }

    /// Validate threshold file fields required by the final-response scorer.
    pub fn validate_final_response(&self) -> AnyResult<()> {
        anyhow::ensure!(
            self.schema_version == FINAL_RESPONSE_THRESHOLDS_SCHEMA_VERSION,
            "unsupported final-response thresholds schema '{}'",
            self.schema_version
        );
        for label in FINAL_RESPONSE_EXPECTED_LABELS {
            let threshold = self
                .labels
                .get(label)
                .ok_or_else(|| anyhow::anyhow!("missing final-response threshold for '{label}'"))?;
            anyhow::ensure!(
                threshold.advisory_min_confidence.is_finite()
                    && threshold.enforce_min_confidence.is_finite(),
                "final-response thresholds for '{label}' must be finite"
            );
        }
        Ok(())
    }

    /// Return thresholds for a label, defaulting to shadow-only.
    pub fn for_label(&self, label: &ToolCallClass) -> LabelThreshold {
        let key = match label {
            ToolCallClass::Valid => "valid",
            ToolCallClass::WrongToolSemantic => "wrong_tool_semantic",
            ToolCallClass::WrongArgumentsSemantic => "wrong_arguments_semantic",
            ToolCallClass::ToolNotNeeded => "tool_not_needed",
            ToolCallClass::NeedsClarification => "needs_clarification",
            ToolCallClass::DeterministicInvalid => "deterministic_invalid",
            ToolCallClass::Unknown(_) => "",
        };

        self.labels
            .get(key)
            .cloned()
            .unwrap_or_else(|| LabelThreshold {
                action: "shadow_only".to_string(),
                advisory_min_confidence: 1.01,
                enforce_min_confidence: 1.01,
            })
    }

    /// Return thresholds for a final-response label, defaulting to shadow-only.
    pub fn for_final_response_label(&self, label: &FinalResponseClass) -> LabelThreshold {
        let key = match label {
            FinalResponseClass::ValidFinalResponse => "valid_final_response",
            FinalResponseClass::MissingToolFact => "missing_tool_fact",
            FinalResponseClass::ContradictsToolResult => "contradicts_tool_result",
            FinalResponseClass::UnsupportedClaim => "unsupported_claim",
            FinalResponseClass::FailedToAcknowledgeDataGap => "failed_to_acknowledge_data_gap",
            FinalResponseClass::Unknown(_) => "",
        };

        self.labels
            .get(key)
            .cloned()
            .unwrap_or_else(|| LabelThreshold {
                action: "shadow_only".to_string(),
                advisory_min_confidence: 1.01,
                enforce_min_confidence: 1.01,
            })
    }
}

/// Label mapping file.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct LabelsFile {
    /// Published label mode.
    pub label_mode: String,
    /// Production labels in model-output order.
    pub labels: Vec<String>,
    /// Label-to-index mapping.
    pub label2id: HashMap<String, usize>,
    /// Index-to-label mapping. JSON object keys are decimal strings.
    pub id2label: HashMap<String, String>,
}

impl LabelsFile {
    /// Validate that labels match the production classifier order.
    pub fn validate(&self) -> AnyResult<()> {
        self.validate_tool_call()
    }

    /// Validate that labels match a supported tool-call production order.
    pub fn validate_tool_call(&self) -> AnyResult<()> {
        validate_supported_label_order(&self.labels)?;
        self.validate_indices()
    }

    /// Validate that labels match the final-response production order.
    pub fn validate_final_response(&self) -> AnyResult<()> {
        validate_final_response_label_order(&self.labels)?;
        self.validate_indices()
    }

    fn validate_indices(&self) -> AnyResult<()> {
        for (index, label) in self.labels.iter().enumerate() {
            anyhow::ensure!(
                self.label2id.get(label.as_str()) == Some(&index),
                "classifier label2id mismatch for '{label}'"
            );
            anyhow::ensure!(
                self.id2label.get(&index.to_string()).map(String::as_str) == Some(label.as_str()),
                "classifier id2label mismatch for index {index}"
            );
        }
        Ok(())
    }
}

/// Parsed classifier artifact files from a local directory.
#[derive(Debug, Clone)]
pub struct ClassifierArtifact {
    /// Artifact directory.
    pub dir: PathBuf,
    /// Parsed manifest.
    pub manifest: ArtifactManifest,
    /// Parsed labels file.
    pub labels: LabelsFile,
    /// Parsed thresholds file.
    pub thresholds: Thresholds,
}

impl ClassifierArtifact {
    /// Load and validate artifact metadata from a local directory.
    pub fn from_dir(path: impl AsRef<Path>) -> AnyResult<Self> {
        let dir = path.as_ref().to_path_buf();
        let manifest: ArtifactManifest = read_json(&dir.join("artifact_manifest.json"))?;
        manifest.validate()?;
        let labels: LabelsFile = read_json(&dir.join("labels.json"))?;
        labels.validate()?;
        let thresholds: Thresholds = read_json(&dir.join("thresholds.json"))?;
        thresholds.validate_tool_call_for_labels(&labels.labels)?;
        Ok(Self {
            dir,
            manifest,
            labels,
            thresholds,
        })
    }

    /// Return the ONNX model path for the requested model kind.
    pub fn model_path(&self, kind: ClassifierModelKind) -> PathBuf {
        let file = match kind {
            ClassifierModelKind::Quantized => &self.manifest.quantized_onnx_file,
            ClassifierModelKind::Full => &self.manifest.onnx_file,
        };
        self.dir.join(file)
    }
}

/// Parsed final-response classifier artifact files from a local directory.
#[derive(Debug, Clone)]
pub struct FinalResponseClassifierArtifact {
    /// Artifact directory.
    pub dir: PathBuf,
    /// Parsed manifest.
    pub manifest: ArtifactManifest,
    /// Parsed labels file.
    pub labels: LabelsFile,
    /// Parsed thresholds file.
    pub thresholds: Thresholds,
}

impl FinalResponseClassifierArtifact {
    /// Load and validate final-response artifact metadata from a local directory.
    pub fn from_dir(path: impl AsRef<Path>) -> AnyResult<Self> {
        let dir = path.as_ref().to_path_buf();
        let manifest: ArtifactManifest = read_json(&dir.join("artifact_manifest.json"))?;
        manifest.validate_final_response()?;
        let labels: LabelsFile = read_json(&dir.join("labels.json"))?;
        labels.validate_final_response()?;
        let thresholds: Thresholds = read_json(&dir.join("thresholds.json"))?;
        thresholds.validate_final_response()?;
        Ok(Self {
            dir,
            manifest,
            labels,
            thresholds,
        })
    }

    /// Return the ONNX model path for the requested model kind.
    pub fn model_path(&self, kind: ClassifierModelKind) -> PathBuf {
        let file = match kind {
            ClassifierModelKind::Quantized => &self.manifest.quantized_onnx_file,
            ClassifierModelKind::Full => &self.manifest.onnx_file,
        };
        self.dir.join(file)
    }
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> AnyResult<T> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read classifier artifact {}", path.display()))?;
    serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse classifier artifact {}", path.display()))
}

fn validate_supported_label_order(labels: &[String]) -> AnyResult<()> {
    if labels_match(labels, &LEGACY_EXPECTED_LABELS) || labels_match(labels, &EXPECTED_LABELS) {
        return Ok(());
    }
    let expected = EXPECTED_LABELS
        .iter()
        .map(|label| label.to_string())
        .collect::<Vec<_>>();
    let legacy_expected = LEGACY_EXPECTED_LABELS
        .iter()
        .map(|label| label.to_string())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        false,
        "classifier labels must be {:?} or {:?}, got {:?}",
        legacy_expected,
        expected,
        labels
    );
    Ok(())
}

fn validate_final_response_label_order(labels: &[String]) -> AnyResult<()> {
    if labels_match(labels, &FINAL_RESPONSE_EXPECTED_LABELS) {
        return Ok(());
    }
    let expected = FINAL_RESPONSE_EXPECTED_LABELS
        .iter()
        .map(|label| label.to_string())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        false,
        "final-response labels must be {:?}, got {:?}",
        expected,
        labels
    );
    Ok(())
}

fn validate_tool_call_serializer_pair(input_schema: &str, serializer: &str) -> AnyResult<()> {
    match (input_schema, serializer) {
        (EXPECTED_INPUT_SCHEMA_VERSION, EXPECTED_SERIALIZER) => Ok(()),
        (NEXT_INPUT_SCHEMA_VERSION, NEXT_SERIALIZER) => Ok(()),
        _ => {
            anyhow::bail!(
                "unsupported classifier input schema '{}' with serializer '{}'",
                input_schema,
                serializer
            )
        }
    }
}

fn labels_match<const N: usize>(labels: &[String], expected: &[&str; N]) -> bool {
    labels.len() == expected.len()
        && labels
            .iter()
            .zip(expected.iter())
            .all(|(actual, expected)| actual == expected)
}
