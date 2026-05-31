use crate::guardrails::{
    ClassifierModelKind, DEFAULT_CLASSIFIER_REPO, DEFAULT_CLASSIFIER_REVISION,
    DEFAULT_FINAL_RESPONSE_CLASSIFIER_REPO, DEFAULT_FINAL_RESPONSE_CLASSIFIER_REVISION,
};
use std::path::PathBuf;

/// Classifier artifact family available for download.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassifierArtifactKind {
    /// Tool-call verifier artifact.
    ToolCall,
    /// Final-response verifier artifact.
    FinalResponse,
}

impl ClassifierArtifactKind {
    /// Return the stable artifact name used in CLI output and cache paths.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ToolCall => "tool-call",
            Self::FinalResponse => "final-response",
        }
    }

    /// Return the default Hugging Face repository for this artifact family.
    pub fn default_repo(self) -> &'static str {
        match self {
            Self::ToolCall => DEFAULT_CLASSIFIER_REPO,
            Self::FinalResponse => DEFAULT_FINAL_RESPONSE_CLASSIFIER_REPO,
        }
    }

    /// Return the pinned Hugging Face revision for this artifact family.
    pub fn default_revision(self) -> &'static str {
        match self {
            Self::ToolCall => DEFAULT_CLASSIFIER_REVISION,
            Self::FinalResponse => DEFAULT_FINAL_RESPONSE_CLASSIFIER_REVISION,
        }
    }

    /// Return the legacy eval/training output directory for this artifact family.
    pub fn default_eval_output_dir(self) -> PathBuf {
        match self {
            Self::ToolCall => PathBuf::from("target/classifier-artifacts"),
            Self::FinalResponse => PathBuf::from("target/final-response-classifier-artifacts"),
        }
    }

    /// Return the CLI output key for this artifact family.
    pub fn output_key(self) -> &'static str {
        match self {
            Self::ToolCall => "classifier_dir",
            Self::FinalResponse => "final_response_classifier_dir",
        }
    }
}

/// Plan for downloading an artifact in repository-tree layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassifierDownloadPlan {
    /// Artifact family.
    pub kind: ClassifierArtifactKind,
    /// Hugging Face repository in owner/name form.
    pub repo: String,
    /// Hugging Face revision.
    pub revision: String,
    /// Local output directory. Repository paths like `onnx/` are created below it.
    pub output_dir: PathBuf,
}

impl ClassifierDownloadPlan {
    /// Return the ONNX artifact directory produced by this plan.
    pub fn artifact_dir(&self) -> PathBuf {
        self.output_dir.join("onnx")
    }
}

/// Result of a classifier artifact preparation operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassifierDownloadReport {
    /// Artifact family.
    pub kind: ClassifierArtifactKind,
    /// Hugging Face repository in owner/name form.
    pub repo: String,
    /// Hugging Face revision.
    pub revision: String,
    /// ONNX model kind prepared.
    pub model_kind: ClassifierModelKind,
    /// Runnable local ONNX artifact directory.
    pub artifact_dir: PathBuf,
    /// Whether any file was downloaded during this operation.
    pub downloaded: bool,
    /// Number of optional sidecar files downloaded.
    pub optional_sidecars: usize,
}
