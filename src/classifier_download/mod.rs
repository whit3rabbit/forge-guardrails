//! Hugging Face download helpers for local ONNX classifier artifacts.

/// Hugging Face synchronization download helper logic.
pub mod download;
/// Pre-requisite/required artifact file definitions.
pub mod files;
pub(crate) mod mod_impl;
/// Cache directory path resolution.
pub mod paths;
#[cfg(test)]
mod tests;
/// Artifact downloader request/response types.
pub mod types;

pub use download::download_classifier_artifact_tree;
pub use files::{required_files_for_model, runtime_required_files};
pub use mod_impl::{
    classifier_artifact_needs_download, ensure_classifier_artifact_dir,
    validate_classifier_artifact_dir,
};
pub use paths::{
    default_cached_classifier_artifact_dir, default_classifier_cache_root,
    default_tool_call_classifier_artifact_dir,
};
pub use types::{ClassifierArtifactKind, ClassifierDownloadPlan, ClassifierDownloadReport};
