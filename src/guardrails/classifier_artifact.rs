//! Classifier artifact metadata parsing and validation.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{Context, Result as AnyResult};
use serde::Deserialize;

use crate::guardrails::scoring::ToolCallClass;

/// Default Hugging Face model repository for the tool-call verifier.
pub const DEFAULT_CLASSIFIER_REPO: &str = "cowWhySo/toolcall-verifier-classifier-production";
/// Pinned Hugging Face revision used by the downloader unless overridden.
pub const DEFAULT_CLASSIFIER_REVISION: &str = "24403e1820d5a9b2279b67766e629477dd1577ee";
/// Expected classifier artifact schema version.
pub const EXPECTED_ARTIFACT_SCHEMA_VERSION: &str = "toolcall-verifier-artifact/v1";
/// Expected classifier input schema version.
pub const EXPECTED_INPUT_SCHEMA_VERSION: &str = "toolcall-verifier-input/v1";
/// Expected classifier serializer name.
pub const EXPECTED_SERIALIZER: &str = "serialize_state_v1";
/// Expected classifier thresholds schema version.
pub const EXPECTED_THRESHOLDS_SCHEMA_VERSION: &str = "toolcall-verifier-thresholds/v1";
/// Labels expected by the production ONNX classifier.
pub const EXPECTED_LABELS: [&str; 5] = [
    "valid",
    "wrong_tool_semantic",
    "tool_not_needed",
    "needs_clarification",
    "deterministic_invalid",
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
        anyhow::ensure!(
            self.artifact_schema_version == EXPECTED_ARTIFACT_SCHEMA_VERSION,
            "unsupported classifier artifact schema '{}'",
            self.artifact_schema_version
        );
        anyhow::ensure!(
            self.input_schema_version == EXPECTED_INPUT_SCHEMA_VERSION,
            "unsupported classifier input schema '{}'",
            self.input_schema_version
        );
        anyhow::ensure!(
            self.serializer == EXPECTED_SERIALIZER,
            "unsupported classifier serializer '{}'",
            self.serializer
        );
        if !self.labels.is_empty() {
            validate_label_order(&self.labels)?;
        }
        anyhow::ensure!(
            self.max_length > 0,
            "classifier max_length must be positive"
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
        anyhow::ensure!(
            self.schema_version == EXPECTED_THRESHOLDS_SCHEMA_VERSION,
            "unsupported classifier thresholds schema '{}'",
            self.schema_version
        );
        for label in EXPECTED_LABELS {
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
        Ok(())
    }

    /// Return thresholds for a label, defaulting to shadow-only.
    pub fn for_label(&self, label: &ToolCallClass) -> LabelThreshold {
        let key = match label {
            ToolCallClass::Valid => "valid",
            ToolCallClass::WrongToolSemantic => "wrong_tool_semantic",
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
        validate_label_order(&self.labels)?;
        for (index, label) in EXPECTED_LABELS.iter().enumerate() {
            anyhow::ensure!(
                self.label2id.get(*label) == Some(&index),
                "classifier label2id mismatch for '{label}'"
            );
            anyhow::ensure!(
                self.id2label.get(&index.to_string()).map(String::as_str) == Some(*label),
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
        thresholds.validate()?;
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

fn validate_label_order(labels: &[String]) -> AnyResult<()> {
    let expected = EXPECTED_LABELS
        .iter()
        .map(|label| label.to_string())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        labels == expected,
        "classifier labels must be {:?}, got {:?}",
        expected,
        labels
    );
    Ok(())
}
